import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [inputValue, setInputValue] = useState('');
  const [lastBooking, setLastBooking] = useState(null);
  const [showBookingConfirmation, setShowBookingConfirmation] = useState(false);
  
  // Pre-populate with enhanced welcome message
  const [chatHistory, setChatHistory] = useState([
    { 
      sender: 'agent', 
      text: "🏥 Welcome to GP Assistant - Available 24/7\n\nI can help you:\n• Book GP appointments instantly\n• Cancel appointments\n• Check appointment status\n\nNo more waiting on hold at 9am! How can I help you today?",
      isWelcome: true 
    }
  ]);

  const [appointments, setAppointments] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [databaseUpdateAnimation, setDatabaseUpdateAnimation] = useState(false);
  const chatWindowRef = useRef(null);

  // Fetch Appointments with animation trigger 
  const fetchAppointments = async (triggerAnimation = false) => {
    try {
      if (triggerAnimation) {
        setDatabaseUpdateAnimation(true);
        setTimeout(() => setDatabaseUpdateAnimation(false), 1000);
      }
      
      const response = await fetch('http://localhost:8000/appointments');
      if (!response.ok) {
        throw new Error('Failed to fetch appointments');
      }
      const data = await response.json();
      setAppointments(data);
    } catch (error) {
      console.error("Failed to fetch appointments:", error);
    }
  };

  const refreshAppointmentSlots = async () => {
    try {
      const response = await fetch('http://localhost:8000/appointments/refresh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!response.ok) {
        throw new Error('Failed to refresh appointment slots');
      }
      await fetchAppointments(true);
      alert('Appointment slots have been refreshed for the next 5 business days!');
    } catch (error) {
      console.error("Failed to refresh appointment slots:", error);
      alert('Failed to refresh appointment slots');
    }
  };

  useEffect(() => {
    fetchAppointments();
    // Poll for updates every 5 seconds
    const interval = setInterval(() => fetchAppointments(false), 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [chatHistory, isLoading]);

  // Parse agent messages for booking confirmations
  const parseBookingConfirmation = (text) => {
    const bookingRefMatch = text.match(/Booking Reference: ([A-Z0-9]+)/);
    const dateTimeMatch = text.match(/Date & Time: ([^\n]+)/);
    const patientMatch = text.match(/Patient: ([^\n]+)/);
    
    if (bookingRefMatch) {
      return {
        reference: bookingRefMatch[1],
        dateTime: dateTimeMatch ? dateTimeMatch[1] : '',
        patient: patientMatch ? patientMatch[1] : ''
      };
    }
    return null;
  };

  const handleSendMessage = async () => {
    const messageToSend = inputValue.trim();
    if (!messageToSend) return;

    // Filter out welcome message from history for API
    const apiHistory = chatHistory
      .filter(msg => !msg.isWelcome)
      .map(msg => ({
        role: msg.sender === 'user' ? 'user' : 'assistant',
        content: msg.text
      }));

    const newUserMessage = { sender: 'user', text: messageToSend };
    setChatHistory(prev => [...prev, newUserMessage]);
    setIsLoading(true);
    setInputValue('');

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: messageToSend, history: apiHistory }),
      });

      if (!response.ok) throw new Error('Network response was not ok');
      
      const data = await response.json();
      
      // Check if this is a booking confirmation
      const bookingInfo = parseBookingConfirmation(data.response);
      if (bookingInfo) {
        setLastBooking(bookingInfo);
        setShowBookingConfirmation(true);
        setTimeout(() => setShowBookingConfirmation(false), 5000);
      }
      
      const newAgentMessage = { 
        sender: 'agent', 
        text: data.response,
        hasBooking: !!bookingInfo 
      };
      setChatHistory(prev => [...prev, newAgentMessage]);

      // Trigger database update animation for booking/cancellation actions
      const lowerCaseMessage = messageToSend.toLowerCase();
      const lowerCaseResponse = data.response.toLowerCase();
      if ((lowerCaseMessage.includes('book') || lowerCaseMessage.includes('cancel')) &&
          (lowerCaseResponse.includes('confirmed') || lowerCaseResponse.includes('cancelled'))) {
        fetchAppointments(true);
      }

    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage = { sender: 'agent', text: 'Sorry, something went wrong with the connection.' };
      setChatHistory(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Format message with special styling for confirmations
  const formatMessage = (text, hasBooking) => {
    if (hasBooking || text.includes('✅')) {
      // Split the message into lines and style accordingly
      const lines = text.split('\n');
      return (
        <div className="booking-confirmation-message">
          {lines.map((line, index) => {
            if (line.includes('✅')) {
              return <div key={index} className="confirmation-header">{line}</div>;
            } else if (line.includes('📋') || line.includes('📅') || line.includes('👤')) {
              return <div key={index} className="confirmation-detail">{line}</div>;
            } else {
              return <div key={index}>{line}</div>;
            }
          })}
        </div>
      );
    }
    return text;
  };

  // Group appointments by date for display
  const groupAppointmentsByDate = (appointments) => {
    const grouped = {};
    appointments.forEach(appt => {
      const date = appt.slot_time.split(' ')[0];
      if (!grouped[date]) {
        grouped[date] = [];
      }
      grouped[date].push(appt);
    });
    return grouped;
  };

  const formatDate = (dateStr) => {
    const date = new Date(dateStr + 'T00:00:00');
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);
    
    if (date.getTime() === today.getTime()) return 'Today';
    if (date.getTime() === tomorrow.getTime()) return 'Tomorrow';
    
    return date.toLocaleDateString('en-GB', { 
      weekday: 'short', 
      day: 'numeric', 
      month: 'short' 
    });
  };

  const formatTime = (timeStr) => {
    const [, time] = timeStr.split(' ');
    const [hours, minutes] = time.split(':');
    return `${hours}:${minutes}`;
  };

  const groupedAppointments = groupAppointmentsByDate(appointments);

  return (
    <div className="main-container">
      {/* Booking Confirmation Popup */}
      {showBookingConfirmation && lastBooking && (
        <div className="booking-popup">
          <div className="booking-popup-content">
            <div className="booking-popup-checkmark">✅</div>
            <h3>Appointment Confirmed!</h3>
            <div className="booking-popup-details">
              <p><strong>Reference:</strong> {lastBooking.reference}</p>
              <p><strong>Date & Time:</strong> {lastBooking.dateTime}</p>
              <p><strong>Patient:</strong> {lastBooking.patient}</p>
            </div>
            <p className="booking-popup-note">Your appointment has been saved to the system</p>
          </div>
        </div>
      )}

      <div className="App">
        <header className="App-header">
          <h1>🏥 GP Assistant</h1>
          <span className="availability-badge">Available 24/7</span>
        </header>
        <div className="chat-window" ref={chatWindowRef}>
          {chatHistory.map((msg, index) => (
            <div key={index} className={`message-container ${msg.sender}`}>
              <div className={`message-bubble ${msg.sender}-bubble ${msg.hasBooking ? 'booking-bubble' : ''}`}>
                {msg.isWelcome ? (
                  <div className="welcome-message">{msg.text}</div>
                ) : (
                  formatMessage(msg.text, msg.hasBooking)
                )}
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="message-container agent">
              <div className="message-bubble agent-bubble loading-bubble">
                <span>.</span><span>.</span><span>.</span>
              </div>
            </div>
          )}
        </div>
        <div className="input-area">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            placeholder="Type your message here..."
            disabled={isLoading}
          />
          <button onClick={handleSendMessage} disabled={isLoading}>
            Send
          </button>
        </div>
      </div>
      
      <div className={`appointment-status ${databaseUpdateAnimation ? 'database-updating' : ''}`}>
        <div className="database-header">
          <h2>📊 Live Database View</h2>
          <div className="database-status">
            <span className="status-dot"></span>
            <span>Real-time sync</span>
          </div>
        </div>
        
        <div className="appointments-grid">
          {Object.keys(groupedAppointments).length > 0 ? (
            Object.entries(groupedAppointments)
              .sort(([a], [b]) => a.localeCompare(b))
              .slice(0, 5) // Show first 5 days
              .map(([date, appts]) => (
                <div key={date} className="date-group">
                  <h3 className="date-header">{formatDate(date)}</h3>
                  <div className="appointment-slots">
                    {appts.slice(0, 18).map(appt => (
                      <div 
                        key={appt.id} 
                        className={`appointment-slot ${appt.is_booked ? 'booked' : 'available'}`}
                      >
                        <div className="slot-time">{formatTime(appt.slot_time)}</div>
                        {appt.is_booked ? (
                          <div className="slot-details">
                            <div className="patient-name">{appt.patient_name}</div>
                            {appt.booking_reference && (
                              <div className="booking-ref">Ref: {appt.booking_reference}</div>
                            )}
                          </div>
                        ) : (
                          <div className="slot-status">Available</div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ))
          ) : (
            <div className="no-appointments">No appointments in database</div>
          )}
        </div>
        
        <div className="database-footer">
          <button onClick={() => fetchAppointments(true)} className="refresh-button">
            🔄 Refresh Database
          </button>
          <button onClick={refreshAppointmentSlots} className="refresh-button generate-button" style={{marginTop: '10px'}}>
            ✨ Generate New Slots
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;