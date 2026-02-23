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


function ProfilePage({ token, onBack }) {
  const [profile, setProfile] = useState(null);
  const [formData, setFormData] = useState({ username: '', password: '', full_name: '', address: '' });
  const [status, setStatus] = useState({ message: '', type: '' });

  const fetchProfile = async () => {
    try {
      const response = await fetch('http://localhost:8000/users/me', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setProfile(data);
        setFormData({ username: data.username, full_name: data.full_name, address: data.address, password: '' });
      }
    } catch (err) {
      console.error("Failed to load profile", err);
    }
  };

  useEffect(() => {
    fetchProfile();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  const handleUpdate = async (e) => {
    e.preventDefault();
    setStatus({ message: 'Updating...', type: 'info' });
    
    // Only send fields that have been changed/filled out
    const payload = {};
    if (formData.username !== profile.username) payload.username = formData.username;
    if (formData.full_name !== profile.full_name) payload.full_name = formData.full_name;
    if (formData.address !== profile.address) payload.address = formData.address;
    if (formData.password) payload.password = formData.password;

    if (Object.keys(payload).length === 0) {
      setStatus({ message: 'No changes made.', type: 'info' });
      return;
    }

    try {
      const response = await fetch('http://localhost:8000/users/me', {
        method: 'PUT',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(payload)
      });
      
      if (!response.ok) throw new Error('Update failed');
      setStatus({ message: 'Profile updated successfully!', type: 'success' });
      fetchProfile(); // Refresh data
      setFormData(prev => ({ ...prev, password: '' })); // Clear password field
    } catch (err) {
      setStatus({ message: 'Failed to update. Username might be taken.', type: 'error' });
    }
  };

  if (!profile) return <div className="profile-container">Loading profile...</div>;

  return (
    <div className="profile-container">
      <div className="profile-header">
        <h2>👤 My Profile</h2>
        <button onClick={onBack} className="back-btn">← Back to Chat</button>
      </div>

      <div className="profile-grid">
        {/* Medical Info Section */}
        <div className="profile-section medical-info">
          <h3>🩺 Medical Information</h3>
          <div className="info-card">
            <h4>Repeat Prescriptions</h4>
            <p>{profile.prescriptions}</p>
          </div>
          <div className="info-card">
            <h4>Important Notes</h4>
            <p>{profile.medical_notes}</p>
          </div>
          <div className="info-card">
            <h4>Date of Birth</h4>
            <p>{profile.dob}</p>
          </div>
        </div>

        {/* Account Settings Section */}
        <div className="profile-section account-settings">
          <h3>⚙️ Account Settings</h3>
          {status.message && (
            <div className={`status-banner ${status.type}`}>
              {status.message}
            </div>
          )}
          <form onSubmit={handleUpdate} className="profile-form">
            <label>Full Name</label>
            <input value={formData.full_name} onChange={e => setFormData({...formData, full_name: e.target.value})} required />
            
            <label>Home Address</label>
            <input value={formData.address} onChange={e => setFormData({...formData, address: e.target.value})} required />
            
            <label>Username</label>
            <input value={formData.username} onChange={e => setFormData({...formData, username: e.target.value})} required />
            
            <label>Change Password (leave blank to keep current)</label>
            <input type="password" placeholder="New Password" value={formData.password} onChange={e => setFormData({...formData, password: e.target.value})} />
            
            <button type="submit" className="save-btn">Save Changes</button>
          </form>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [token, setToken] = useState(sessionStorage.getItem('token'));
  const [view, setView] = useState(sessionStorage.getItem('token') ? 'dashboard' : 'login');
  
  const [inputValue, setInputValue] = useState('');
  const [showBookingConfirmation, setShowBookingConfirmation] = useState(false);
  const [appointments, setAppointments] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  // track the emergency lockout state 
  const [isEmergencyLock, setIsEmergencyLock] = useState(false);
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
    sessionStorage.setItem('token', accessToken);
    setToken(accessToken);
    setView('dashboard');
    fetchAppointments(accessToken);
    fetchChatHistory(accessToken);
  };

  // Clear auth token and deeply reset all frontend state
  const handleLogout = () => {
    sessionStorage.removeItem('token');
    setToken(null);
    setView('login');
    
    // Completely clear previous user's data
    setChatHistory([{ 
      sender: 'agent', 
      text: "🏥 Welcome to GP Assistant.\n\nHow can I help you today?",
      isWelcome: true 
    }]);
    setAppointments([]);
    setInputValue('');
    setIsEmergencyLock(false);
    setShowBookingConfirmation(false);
  };

  // Fetch saved chat logs from the database
  const fetchChatHistory = async (authToken = token) => {
    if (!authToken) return;
    try {
      const response = await fetch('http://localhost:8000/chat/history', {
        headers: { 'Authorization': `Bearer ${authToken}` }
      });
      if (response.ok) {
        const data = await response.json();
        if (data.length > 0) {
          const formattedHistory = data.map(msg => ({
            sender: msg.role === 'user' ? 'user' : 'agent',
            text: msg.content
          }));
          // Prepend the welcome message so it's always at the top
          setChatHistory([
            { 
              sender: 'agent', 
              text: "🏥 Welcome back to GP Assistant.",
              isWelcome: true 
            },
            ...formattedHistory
          ]);
        }
      }
    } catch (error) {
      console.error("Failed to fetch chat history:", error);
    }
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
      fetchChatHistory(token);
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
    const lowerText = text.toLowerCase();
    const isConfirmed = lowerText.includes('successfully booked') || lowerText.includes('appointment confirmed');
    return isConfirmed ? true : false;
  };

  const handleSendMessage = async () => {
    // Prevent API calls if locked 
    if (isEmergencyLock) return;

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


      // Check for the emergency status trigger
      if (data.status === 'emergency') {
        setIsEmergencyLock(true);
      }

      
      // Check if this is a booking confirmation
      const bookingInfo = parseBookingConfirmation(data.response);
      if (bookingInfo) {
        setShowBookingConfirmation(true);
        setTimeout(() => setShowBookingConfirmation(false), 5000);
        fetchAppointments(token); 
      } else if (data.response.toLowerCase().includes('cancelled')) {
        fetchAppointments(token);
      }
      
      const newAgentMessage = { 
        sender: 'agent', 
        text: data.response,
        hasBooking: !!bookingInfo,
        source: data.source 
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
          {appointments.filter(a => a.id && new Date(a.slot_time) >= new Date(Date.now() - 48 * 60 * 60 * 1000)).length > 0 ? (
            appointments
              .filter(a => a.id && new Date(a.slot_time) >= new Date(Date.now() - 48 * 60 * 60 * 1000))
              .sort((a, b) => new Date(a.slot_time) - new Date(b.slot_time)) // Sort chronologically
              .map((appt) => {
                const apptDate = new Date(appt.slot_time);
                const isPast = apptDate < new Date(); // Check if the appointment has already happened
                
                return (
                  <div 
                    key={appt.id} 
                    className="appointment-card" 
                    // Visually dim the card if it's in the past
                    style={{ opacity: isPast ? 0.6 : 1, backgroundColor: isPast ? '#f7fafc' : 'white' }} 
                  >
                    <div className="appt-time">
                      {apptDate.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' })}
                      <br/>
                      @ {apptDate.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })}
                    </div>
                    <div 
                      className="appt-status" 
                      style={{
                        marginTop: '8px', 
                        fontWeight: 'bold',
                        // Change color based on past/future status
                        color: isPast ? '#718096' : '#38B2AC' 
                      }}
                    >
                      {isPast ? '✓ Completed' : 'Confirmed'}
                    </div>
                  </div>
                );
              })
          ) : (
            // Empty state when no valid appointments exist
            <div className="empty-state">
              Your scheduled appointments will appear here.
            </div>
          )}
        </div>
        
        <button onClick={handleLogout} className="logout-btn">
          Log Out
        </button>
      </div>

      {/* Main chat interface or Profile Page */}
      <div className="chat-container">
        
        {/* Top Navigation Bar */}
        <div className="top-nav-bar">
          {view === 'dashboard' ? (
            <button onClick={() => setView('profile')} className="profile-icon-btn" title="View Profile">
              👤
            </button>
          ) : (
            <div style={{height: '40px'}}></div> // Spacer when in profile view
          )}
        </div>

        {view === 'profile' ? (
          <ProfilePage token={token} onBack={() => setView('dashboard')} />
        ) : (
          <>

        
        {/* Medical Source Pop-up Card */}
        
        <div className="chat-history" ref={chatWindowRef}>
          {chatHistory.map((msg, index) => (
            <div key={index} className={`message ${msg.sender === 'user' ? 'user' : 'assistant'}`}>
              
              {/* Wrapper to stack the bubble and the source card */}
              <div className="message-content-wrapper">
                <div className="message-bubble">
                  {msg.text}
                </div>
                
                {/* NEW: Inline Medical Source Card */}
                {msg.source && (
                  <div className="inline-medical-source">
                    <div className="inline-source-header">
                      <span className="nhs-logo-text">NHS</span> Verified Info
                    </div>
                    <div className="inline-source-body">
                      <strong>{msg.source.condition}</strong>
                      <p>"{msg.source.text}"</p>
                      <a href={msg.source.url} target="_blank" rel="noopener noreferrer">
                        Read full guidance on NHS.uk →
                      </a>
                    </div>
                  </div>
                )}
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
            // Update placeholder to explain why it's locked
            placeholder={isEmergencyLock ? "Chat disabled for your safety." : "Type your message here..."}
            // Hard lock the input field
            disabled={isLoading || isEmergencyLock} 
          />
          <button onClick={handleSendMessage} disabled={isLoading || isEmergencyLock}>
            Send
          </button>
        </div>


        {/* Emergency Hard Lock Overlay */}
        {isEmergencyLock && (
          <div className="emergency-alert">
            <h3>
              <span>🛑</span> Medical Assessment Required
            </h3>
            <p className="emergency-description">
              Based on the symptoms described, this automated assistant has been locked for your safety. Please seek human medical advice immediately.
            </p>
            
            <div className="emergency-options-grid">
              <div className="emergency-option-card nhs-111">
                <h4>📞 NHS 111</h4>
                <p>
                  Call <strong>111</strong> for free, 24/7 urgent medical advice. They can direct you to the most appropriate local service.
                </p>
              </div>
              
              <div className="emergency-option-card gp-contact">
                <h4>🏥 Contact Your GP</h4>
                <p>
                  Call our surgery directly at <strong>01234 567 890</strong> to speak with a receptionist.
                </p>
              </div>
            </div>

            <p className="emergency-warning-text">
              If this is a life-threatening emergency, please call <strong>999</strong> immediately.
            </p>
            
            <button
              className="emergency-reset-btn"
              onClick={() => {
                setIsEmergencyLock(false);
                setChatHistory([{ sender: 'agent', text: "Session reset. How can I help you today?", isWelcome: true }]);
              }}
            >
              I understand, reset session
            </button>
          </div>
        )}
          </>
        )}
      </div>
      
      {/* Booking confirmation popup toast */}
      {showBookingConfirmation && (
        <div style={{
          position: 'absolute', top: '30px', right: '30px', 
          background: '#38B2AC', color: 'white', padding: '16px 24px', 
          borderRadius: '16px', boxShadow: '0 10px 25px -5px rgba(0,0,0,0.2)',
          fontWeight: '600', animation: 'slideUp 0.5s ease-out', zIndex: 100
        }}>
          ✅ Booking Confirmed!
        </div>
      )}

    </div>
  );
}

export default App;