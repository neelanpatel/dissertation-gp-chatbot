// frontend/gp-chatbot-ui/src/App.js
import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [inputValue, setInputValue] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [appointments, setAppointments] = useState([]);
  const [isLoading, setIsLoading] = useState(false); // <-- NEW STATE
  const chatWindowRef = useRef(null);

  // --- Fetch Appointments on Load and for Refreshing ---
  const fetchAppointments = async () => {
    try {
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

  useEffect(() => {
    fetchAppointments();
  }, []);
  // --- End of Fetch Appointments ---

  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [chatHistory, isLoading]); // <-- UPDATED DEPENDENCY

  const handleSendMessage = async () => {
    const messageToSend = inputValue.trim();
    if (!messageToSend) return;

    const apiHistory = chatHistory.map(msg => ({
      role: msg.sender === 'user' ? 'user' : 'assistant',
      content: msg.text
    }));

    const newUserMessage = { sender: 'user', text: messageToSend };
    setChatHistory(prev => [...prev, newUserMessage]);
    setIsLoading(true); // <-- SHOW LOADING INDICATOR
    setInputValue('');

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: messageToSend, history: apiHistory }),
      });

      if (!response.ok) throw new Error('Network response was not ok');
      
      const data = await response.json();
      
      const newAgentMessage = { sender: 'agent', text: data.response };
      setChatHistory(prev => [...prev, newAgentMessage]); // Add agent response

      const lowerCaseMessage = messageToSend.toLowerCase();
      if (lowerCaseMessage.includes('book') || lowerCaseMessage.includes('cancel')) {
        setTimeout(() => fetchAppointments(), 500);
      }

    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage = { sender: 'agent', text: 'Sorry, something went wrong with the connection.' };
      setChatHistory(prev => [...prev, errorMessage]); // Add error response
    } finally {
      setIsLoading(false); // <-- HIDE LOADING INDICATOR
    }
  };

  return (
    <div className="main-container">
      <div className="App">
        <header className="App-header">
          <h1>GP Assistant</h1>
        </header>
        <div className="chat-window" ref={chatWindowRef}>
          {chatHistory.map((msg, index) => (
            <div key={index} className={`message-container ${msg.sender}`}>
              <div className={`message-bubble ${msg.sender}-bubble`}>
                {msg.text}
              </div>
            </div>
          ))}
          
          {/* --- NEW LOADING INDICATOR --- */}
          {isLoading && (
            <div className="message-container agent">
              <div className="message-bubble agent-bubble loading-bubble">
                <span>.</span><span>.</span><span>.</span>
              </div>
            </div>
          )}
          {/* --- END OF LOADING INDICATOR --- */}

        </div>
        <div className="input-area">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            placeholder="Type your message here..."
            disabled={isLoading} // <-- Disable input while loading
          />
          <button onClick={handleSendMessage} disabled={isLoading}> {/* <-- Disable button while loading */}
            Send
          </button>
        </div>
      </div>
      
      <div className="appointment-status">
        <h2>Database: Appointments</h2>
        <button onClick={fetchAppointments} className="refresh-button">Refresh</button>
        <ul>
          {appointments.length > 0 ? (
            appointments.map(appt => (
              <li key={appt.id} className={appt.is_booked ? 'booked' : 'available'}>
                ID {appt.id}: {new Date(appt.slot_time).toLocaleString()} - 
                <strong>{appt.is_booked ? ` Booked by ${appt.patient_name}` : ' Available'}</strong>
              </li>
            ))
          ) : (
            <li>No appointments found in the database.</li>
          )}
        </ul>
      </div>
    </div>
  );
}

export default App;