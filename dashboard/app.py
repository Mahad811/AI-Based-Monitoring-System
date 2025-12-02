"""
Flask Dashboard Application
Real-time monitoring interface for nursing staff
"""

from flask import Flask, render_template, jsonify
from datetime import datetime


def create_app(config=None):
    """
    Create and configure Flask app
    
    Args:
        config: Application configuration
        
    Returns:
        Flask app instance
    """
    app = Flask(__name__)
    
    if config:
        app.config.update(config)
    
    # Store recent alerts in memory (in production, use proper database)
    app.alerts = []
    
    @app.route('/')
    def index():
        """Main dashboard page"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vital Guardian - Patient Monitoring</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                .header {
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                }
                .alert-container {
                    margin-top: 20px;
                }
                .alert {
                    background-color: white;
                    padding: 15px;
                    margin: 10px 0;
                    border-left: 5px solid #3498db;
                    border-radius: 3px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .alert.critical {
                    border-left-color: #e74c3c;
                }
                .alert.high {
                    border-left-color: #e67e22;
                }
                .alert.medium {
                    border-left-color: #f39c12;
                }
                .alert.low {
                    border-left-color: #3498db;
                }
                .status {
                    display: inline-block;
                    padding: 5px 10px;
                    border-radius: 3px;
                    color: white;
                    font-weight: bold;
                    margin-right: 10px;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Vital Guardian</h1>
                <p>AI-Powered Patient Monitoring System</p>
            </div>
            
            <div class="alert-container">
                <h2>Recent Alerts</h2>
                <div id="alerts">
                    <p>No alerts. System monitoring...</p>
                </div>
            </div>
            
            <script>
                // Poll for new alerts every 2 seconds
                setInterval(function() {
                    fetch('/api/alerts')
                        .then(response => response.json())
                        .then(data => {
                            updateAlerts(data.alerts);
                        });
                }, 2000);
                
                function updateAlerts(alerts) {
                    const container = document.getElementById('alerts');
                    if (alerts.length === 0) {
                        container.innerHTML = '<p>No alerts. System monitoring...</p>';
                        return;
                    }
                    
                    container.innerHTML = alerts.map(alert => `
                        <div class="alert ${alert.priority}">
                            <span class="status" style="background-color: ${getPriorityColor(alert.priority)}">
                                ${alert.priority.toUpperCase()}
                            </span>
                            <strong>${alert.timestamp}</strong>
                            <p>${alert.description}</p>
                        </div>
                    `).join('');
                }
                
                function getPriorityColor(priority) {
                    const colors = {
                        'critical': '#e74c3c',
                        'high': '#e67e22',
                        'medium': '#f39c12',
                        'low': '#3498db'
                    };
                    return colors[priority] || '#3498db';
                }
            </script>
        </body>
        </html>
        """
    
    @app.route('/api/alerts')
    def get_alerts():
        """API endpoint for alerts"""
        return jsonify({
            'alerts': app.alerts[-10:],  # Return last 10 alerts
            'count': len(app.alerts)
        })
    
    @app.route('/api/alert/add', methods=['POST'])
    def add_alert():
        """API endpoint to add new alert"""
        from flask import request
        alert_data = request.json
        app.alerts.append(alert_data)
        return jsonify({'status': 'success'})
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)

