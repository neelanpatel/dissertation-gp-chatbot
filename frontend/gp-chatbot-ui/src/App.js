import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';


function ForgotPasswordForm({ onBack }) {
  const [step, setStep] = React.useState('verify'); // 'verify' or 'reset'
  const [username, setUsername] = React.useState('');
  const [dob, setDob] = React.useState('');
  const [newPassword, setNewPassword] = React.useState('');
  const [confirmPassword, setConfirmPassword] = React.useState('');
  const [error, setError] = React.useState('');
  const [isLoading, setIsLoading] = React.useState(false);

  const handleVerify = async (e) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/auth/verify-identity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, dob })
      });
      if (!response.ok) {
        setError('No account found with those details. Please check your username and date of birth.');
        return;
      }
      setStep('reset');
    } catch (err) {
      setError('Connection error. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = async (e) => {
    e.preventDefault();
    setError('');
    if (newPassword !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }
    if (newPassword.length < 6) {
      setError('Password must be at least 6 characters.');
      return;
    }
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/auth/reset-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, dob, new_password: newPassword })
      });
      if (!response.ok) {
        setError('Something went wrong. Please try again.');
        return;
      }
      setStep('success');
    } catch (err) {
      setError('Connection error. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  if (step === 'success') {
    return (
      <div className="auth-wrapper">
        <div className="auth-container">
          <h2>✅ Password Reset</h2>
          <p>Your password has been updated successfully.</p>
          <button type="button" onClick={onBack} style={{width:'100%', background:'var(--primary-teal)', color:'white', border:'none', padding:'18px', fontSize:'17px', fontWeight:'600', borderRadius:'16px', cursor:'pointer', marginTop:'10px'}}>
            Back to Login
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="auth-wrapper">
      <div className="auth-container">
        <h2>🔑 Reset Password</h2>
        {step === 'verify' ? (
          <>
            <p>Enter your username and date of birth to verify your identity.</p>
            {error && <div style={{color:'#E53E3E', marginBottom:'15px', fontWeight:'500'}}>{error}</div>}
            <form onSubmit={handleVerify}>
              <input
                placeholder="Username"
                value={username}
                onChange={e => setUsername(e.target.value)}
                required
              />
              <input
                type="date"
                value={dob}
                onChange={e => setDob(e.target.value)}
                required
              />
              <button type="submit" disabled={isLoading}>
                {isLoading ? 'Verifying...' : 'Verify Identity'}
              </button>
            </form>
          </>
        ) : (
          <>
            <p>Identity verified. Please enter your new password.</p>
            {error && <div style={{color:'#E53E3E', marginBottom:'15px', fontWeight:'500'}}>{error}</div>}
            <form onSubmit={handleReset}>
              <input
                type="password"
                placeholder="New Password"
                value={newPassword}
                onChange={e => setNewPassword(e.target.value)}
                required
              />
              <input
                type="password"
                placeholder="Confirm New Password"
                value={confirmPassword}
                onChange={e => setConfirmPassword(e.target.value)}
                required
              />
              <button type="submit" disabled={isLoading}>
                {isLoading ? 'Resetting...' : 'Reset Password'}
              </button>
            </form>
          </>
        )}
        <div style={{marginTop:'25px'}}>
          <button className="auth-link" onClick={onBack}>← Back to Login</button>
        </div>
      </div>
    </div>
  );
}
// Login form for existing patients
function LoginForm({ onLogin, onSwitchToRegister, onForgotPassword }) {
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
          <button className="auth-link" onClick={onForgotPassword} style={{display:'block', marginBottom:'12px'}}>Forgot password?</button>
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

function AppointmentPicker({ slots, onBook }) {
  const [selectedDate, setSelectedDate] = React.useState(null);
  const dates = Object.keys(slots);

  return (
    <div className="appointment-picker">
      <div className="picker-header">🗓 Select a date</div>
      <div className="picker-dates">
        {dates.map(date => (
          <button
            key={date}
            className={`picker-date-btn ${selectedDate === date ? 'active' : ''}`}
            onClick={() => setSelectedDate(selectedDate === date ? null : date)}
          >
            {slots[date].date.split(',')[0]}
            <span>{slots[date].date.split(',').slice(1).join(',').trim()}</span>
          </button>
        ))}
      </div>
      {selectedDate && (
        <div className="picker-times">
          {slots[selectedDate].times.map(time => (
            <button
              key={time}
              className="picker-time-btn"
              onClick={() => onBook(slots[selectedDate].date, time)}
            >
              {time}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function CancellationPicker({ bookings, onCancel }) {
  return (
    <div className="cancellation-picker">
      <div className="picker-header">🗑 Select appointment to cancel</div>
      <div className="cancel-bookings-list">
        {bookings.map(booking => (
          <button
            key={booking.id}
            className="cancel-booking-card"
            onClick={() => onCancel(booking)}
          >
            <span className="cancel-booking-date">{booking.display_date}</span>
            <span className="cancel-booking-time">{booking.display_time}</span>
            <span className="cancel-booking-ref">{booking.booking_reference}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function ReschedulePicker({ data, token, onSuccess }) {
  const [selectedBooking, setSelectedBooking] = React.useState(null);
  const [selectedDate, setSelectedDate] = React.useState(null);
  const [selectedTime, setSelectedTime] = React.useState(null);
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [error, setError] = React.useState(null);

  const slots = data.available_slots;
  const dates = Object.keys(slots);
  const canConfirm = selectedBooking && selectedDate && selectedTime && !isSubmitting;

  const handleConfirm = async () => {
    if (!canConfirm) return;
    setIsSubmitting(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/appointments/reschedule', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          old_appointment_id: selectedBooking.id,
          requested_day: slots[selectedDate].date.split(',')[0].trim(),
          requested_time: selectedTime
        })
      });
      const result = await response.json();
      if (!response.ok) {
        setError(response.status === 409
          ? 'That slot was just taken — please choose another time.'
          : 'Something went wrong. Please try again.'
        );
        setSelectedDate(null);
        setSelectedTime(null);
        return;
      }
      onSuccess(result.message);
    } catch (e) {
      setError('Connection error. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="reschedule-picker">
      <div className="picker-header">🔄 Reschedule Appointment</div>
      <div className="reschedule-panels">

        {/* Left: current bookings */}
        <div className="reschedule-panel reschedule-panel-left">
          <div className="reschedule-panel-label">Current appointment</div>
          {data.current_bookings.map(booking => (
            <button
              key={booking.id}
              className={`reschedule-current-card ${selectedBooking?.id === booking.id ? 'active' : ''}`}
              onClick={() => { setSelectedBooking(booking); setSelectedDate(null); setSelectedTime(null); }}
            >
              <span className="reschedule-booking-date">{booking.display_date}</span>
              <span className="reschedule-booking-time">{booking.display_time}</span>
            </button>
          ))}
        </div>

        {/* Right: new slot picker */}
        <div className={`reschedule-panel reschedule-panel-right ${!selectedBooking ? 'panel-disabled' : ''}`}>
          <div className="reschedule-panel-label">New time</div>
          {!selectedBooking ? (
            <div className="reschedule-placeholder">Select a current appointment first</div>
          ) : (
            <>
              <div className="picker-dates">
                {dates.map(date => (
                  <button
                    key={date}
                    className={`picker-date-btn ${selectedDate === date ? 'active' : ''}`}
                    onClick={() => { setSelectedDate(selectedDate === date ? null : date); setSelectedTime(null); }}
                  >
                    {slots[date].date.split(',')[0]}
                    <span>{slots[date].date.split(',').slice(1).join(',').trim()}</span>
                  </button>
                ))}
              </div>
              {selectedDate && (
                <div className="picker-times">
                  {slots[selectedDate].times.map(time => (
                    <button
                      key={time}
                      className={`picker-time-btn ${selectedTime === time ? 'active' : ''}`}
                      onClick={() => setSelectedTime(time)}
                    >
                      {time}
                    </button>
                  ))}
                </div>
              )}
            </>
          )}
        </div>

      </div>
      {error && <div className="reschedule-error">{error}</div>}
      <button
        className="reschedule-confirm-btn"
        disabled={!canConfirm}
        onClick={handleConfirm}
      >
        {isSubmitting ? 'Rescheduling...' : 'Confirm Reschedule'}
      </button>
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
            text: msg.content,
            source: msg.source || null 
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

  const handleBookFromPicker = (day, time) => {
    // Strip the date portion, keep just the weekday name for the booking message
    const dayName = day.split(',')[0].trim();
    setInputValue(`Book ${dayName} at ${time}`);
    // Remove the picker from the message that spawned it
    setChatHistory(prev => prev.map(msg =>
      msg.available_slots ? { ...msg, available_slots: null } : msg
    ));
    // Use setTimeout to let state settle before sending
    setTimeout(() => handleSendMessage(`Book ${dayName} at ${time}`), 50);
  };

  const handleCancelFromPicker = async (booking) => {
    try {
      const response = await fetch('http://localhost:8000/appointments/cancel-by-id', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ appointment_id: booking.id })
      });
      const result = await response.json();
      setChatHistory(prev => prev.map(msg =>
        msg.cancellation_slots ? { ...msg, cancellation_slots: null } : msg
      ));
      if (response.ok) {
        setChatHistory(prev => [...prev, { sender: 'agent', text: `✅ ${result.message}` }]);
        fetchAppointments(token);
      } else {
        setChatHistory(prev => [...prev, { sender: 'agent', text: 'Sorry, I was unable to cancel that appointment. Please try again.' }]);
      }
    } catch (e) {
      setChatHistory(prev => [...prev, { sender: 'agent', text: 'Connection error. Please try again.' }]);
    }
  };

  const handleRescheduleSuccess = (message) => {
    setChatHistory(prev => prev.map(msg =>
      msg.reschedule_data ? { ...msg, reschedule_data: null } : msg
    ));
    setChatHistory(prev => [...prev, { sender: 'agent', text: `✅ ${message}` }]);
    setShowBookingConfirmation(true);
    setTimeout(() => setShowBookingConfirmation(false), 5000);
    fetchAppointments(token);
  };

  const handleSendMessage = async (overrideMessage) => {
    // Prevent API calls if locked 
    if (isEmergencyLock) return;

    const messageToSend = (overrideMessage || inputValue).trim();
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
        source: data.source,
        available_slots: data.available_slots || null,
        cancellation_slots: data.cancellation_slots || null,
        reschedule_data: data.reschedule_data || null
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
    if (view === 'register') return <RegisterForm onLogin={handleLogin} onSwitchToLogin={() => setView('login')} />;
    if (view === 'forgot') return <ForgotPasswordForm onBack={() => setView('login')} />;
    return <LoginForm onLogin={handleLogin} onSwitchToRegister={() => setView('register')} onForgotPassword={() => setView('forgot')} />;
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
                  {msg.sender === 'user' ? msg.text : <ReactMarkdown>{msg.text}</ReactMarkdown>}
                </div>
                
                {/* Inline Appointment Picker */}
                {msg.available_slots && (
                  <AppointmentPicker
                    slots={msg.available_slots}
                    onBook={handleBookFromPicker}
                  />
                )}

                {/* Inline Cancellation Picker */}
                {msg.cancellation_slots && (
                  <CancellationPicker
                    bookings={msg.cancellation_slots}
                    onCancel={handleCancelFromPicker}
                  />
                )}

                {/* Inline Reschedule Widget */}
                {msg.reschedule_data && (
                  <ReschedulePicker
                    data={msg.reschedule_data}
                    token={token}
                    onSuccess={handleRescheduleSuccess}
                  />
                )}

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
          <textarea
            rows={1}
            value={inputValue}
            onChange={(e) => {
              setInputValue(e.target.value);
              e.target.style.height = 'auto';
              e.target.style.height = `${e.target.scrollHeight}px`;
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
              }
            }}
            placeholder={isEmergencyLock ? "Chat disabled for your safety." : "Type your message here..."}
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