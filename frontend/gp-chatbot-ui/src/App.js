import React, { useState, useEffect, useRef } from 'react';
import './App.css';

// Login form for existing patients
function LoginForm({ onLogin, onSwitchToRegister }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    try {
      const response = await fetch('http://localhost:8000/token', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error('Login failed');
      const data = await response.json();
      onLogin(data.access_token);
    } catch (err) {
      setError('Invalid username or password');
    }
  };

  return (
    <div className="auth-wrapper">
      <div className="auth-container">
        <h2>👋 GP Patient Portal</h2>
        <p>Welcome back. Please log in to access your secure assistant.</p>
        {error && <div style={{color: '#E53E3E', marginBottom: '15px', fontWeight: '500'}}>{error}</div>}
        <form onSubmit={handleSubmit}>
          <input 
            placeholder="Username" 
            value={username} 
            onChange={e => setUsername(e.target.value)} 
            required 
          />
          <input 
            type="password" 
            placeholder="Password" 
            value={password} 
            onChange={e => setPassword(e.target.value)} 
            required 
          />
          <button type="submit">Log In</button>
        </form>
        <div style={{marginTop: '25px'}}>
          <p style={{marginBottom: '5px', fontSize: '0.9rem'}}>New to the practice?</p>
          <button className="auth-link" onClick={onSwitchToRegister}>Register as a new patient</button>
        </div>
      </div>
    </div>
  );
}

// Registration form for new patients
function RegisterForm({ onLogin, onSwitchToLogin }) {
  const [formData, setFormData] = useState({
    username: '', password: '', full_name: '', dob: '', address: ''
  });
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('http://localhost:8000/register', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(formData)
      });
      if (!response.ok) throw new Error('Registration failed');
      const data = await response.json();
      onLogin(data.access_token);
    } catch (err) {
      setError('Registration failed. Username may be taken.');
    }
  };

  return (
    <div className="auth-wrapper">
      <div className="auth-container">
        <h2>📝 New Patient Registration</h2>
        <p>Create your secure account below.</p>
        {error && <div style={{color: '#E53E3E', marginBottom: '15px', fontWeight: '500'}}>{error}</div>}
        <form onSubmit={handleSubmit}>
          <input placeholder="Username" onChange={e => setFormData({...formData, username: e.target.value})} required />
          <input type="password" placeholder="Password" onChange={e => setFormData({...formData, password: e.target.value})} required />
          <input placeholder="Full Name" onChange={e => setFormData({...formData, full_name: e.target.value})} required />
          <input type="date" placeholder="Date of Birth" onChange={e => setFormData({...formData, dob: e.target.value})} required />
          <input placeholder="Address" onChange={e => setFormData({...formData, address: e.target.value})} required />
          <button type="submit">Register</button>
        </form>
        <div style={{marginTop: '25px'}}>
          <button className="auth-link" onClick={onSwitchToLogin}>Return to Log In</button>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [view, setView] = useState('login');
  
  const [inputValue, setInputValue] = useState('');
  const [lastBooking, setLastBooking] = useState(null);
  const [showBookingConfirmation, setShowBookingConfirmation] = useState(false);
  const [appointments, setAppointments] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const chatWindowRef = useRef(null);

  // Pre-populate with standard welcome message
  const [chatHistory, setChatHistory] = useState([
    { 
      sender: 'agent', 
      text: "🏥 Welcome to GP Assistant.\n\nI can help you:\n• Book GP appointments\n• Check your symptoms\n• Manage your bookings\n\nHow can I help you today?",
      isWelcome: true 
    }
  ]);

  // Handle successful login and token storage
  const handleLogin = (accessToken) => {
    localStorage.setItem('token', accessToken);
    setToken(accessToken);
    setView('dashboard');
    fetchAppointments(accessToken);
  };

  // Clear auth token and reset state
  const handleLogout = () => {
    localStorage.removeItem('token');
    setToken(null);
    setView('login');
    setChatHistory([{ 
      sender: 'agent', 
      text: "🏥 Welcome to GP Assistant.\n\nHow can I help you today?",
      isWelcome: true 
    }]);
  };

  // Fetch authenticated user's appointments
  const fetchAppointments = async (authToken = token) => {
    if (!authToken) return;
    try {
      const response = await fetch('http://localhost:8000/appointments', {
        headers: { 'Authorization': `Bearer ${authToken}` }
      });
      if (response.status === 401) {
        handleLogout();
        return;
      }
      const data = await response.json();
      setAppointments(data);
    } catch (error) {
      console.error("Failed to fetch appointments:", error);
    }
  };

  useEffect(() => {
    if (token) {
      fetchAppointments(token);
      // Poll for updates every 10 seconds
      const interval = setInterval(() => fetchAppointments(token), 10000);
      return () => clearInterval(interval);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [chatHistory, isLoading]);

  // Parse agent messages for booking confirmations
  const parseBookingConfirmation = (text) => {
    const bookingRefMatch = text.match(/Ref: ([A-Z0-9]+)/) || text.match(/Reference: ([A-Z0-9]+)/);
    return bookingRefMatch ? { reference: bookingRefMatch[1] } : null;
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
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}` 
        },
        body: JSON.stringify({ message: messageToSend, history: apiHistory }),
      });

      if (response.status === 401) {
        handleLogout();
        return;
      }
      if (!response.ok) throw new Error('Network response was not ok');
      
      const data = await response.json();
      
      // Check if this is a booking confirmation
      const bookingInfo = parseBookingConfirmation(data.response);
      if (bookingInfo) {
        setLastBooking(bookingInfo);
        setShowBookingConfirmation(true);
        setTimeout(() => setShowBookingConfirmation(false), 5000);
        fetchAppointments(token); 
      } else if (data.response.toLowerCase().includes('cancelled')) {
        fetchAppointments(token);
      }
      
      const newAgentMessage = { 
        sender: 'agent', 
        text: data.response,
        hasBooking: !!bookingInfo 
      };
      setChatHistory(prev => [...prev, newAgentMessage]);

    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage = { sender: 'agent', text: 'Sorry, I lost connection. Please try again.' };
      setChatHistory(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  if (!token) {
    return view === 'register' 
      ? <RegisterForm onLogin={handleLogin} onSwitchToLogin={() => setView('login')} />
      : <LoginForm onLogin={handleLogin} onSwitchToRegister={() => setView('register')} />;
  }

  return (
    <div className="app-container">
      
      {/* Sidebar showing scheduled appointments */}
      <div className="sidebar">
        <div className="sidebar-header">
          <h3>📅 My Appointments</h3>
        </div>
        
        <div className="appointment-list">
          {appointments.filter(a => a.booking_reference).length > 0 ? (
            appointments
              .filter(a => a.booking_reference)
              .map((appt) => (
                <div key={appt.id} className="appointment-card">
                  <div className="appt-time">
                    {new Date(appt.slot_time).toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' })}
                    <br/>
                    @ {new Date(appt.slot_time).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })}
                  </div>
                  <div className="appt-ref">
                    Ref: {appt.booking_reference}
                  </div>
                  <div className="appt-status">
                    Confirmed
                  </div>
                </div>
              ))
          ) : (
            // Empty state when no appointments exist
            <div className="empty-state">
              Your scheduled appointments will appear here.
            </div>
          )}
        </div>
        
        <button onClick={handleLogout} className="logout-btn">
          Log Out
        </button>
      </div>

      {/* Main chat interface */}
      <div className="chat-container">
        <div className="chat-history" ref={chatWindowRef}>
          {chatHistory.map((msg, index) => (
            <div key={index} className={`message ${msg.sender === 'user' ? 'user' : 'assistant'}`}>
              <div className="message-bubble">
                {msg.text}
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="message assistant">
              <div className="message-bubble typing-indicator">
                ...
              </div>
            </div>
          )}
        </div>
        
        {/* Input area */}
        <div className="chat-input-area">
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
      
      {/* Booking confirmation popup toast */}
      {showBookingConfirmation && lastBooking && (
        <div style={{
          position: 'absolute', top: '30px', right: '30px', 
          background: '#38B2AC', color: 'white', padding: '16px 24px', 
          borderRadius: '16px', boxShadow: '0 10px 25px -5px rgba(0,0,0,0.2)',
          fontWeight: '600', animation: 'slideUp 0.5s ease-out', zIndex: 100
        }}>
          ✅ Booking Confirmed! ({lastBooking.reference})
        </div>
      )}

    </div>
  );
}

export default App;