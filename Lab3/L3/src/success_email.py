import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from airflow.hooks.base import BaseHook


def send_success_email(**kwargs):
    """
    Send success notification email after DAG completion.
    
    Setup required:
    1. Create Airflow connection 'email_smtp' with:
       - Conn Type: Generic
       - Login: your_email@gmail.com
       - Password: your_app_password
    
    2. Enable Gmail App Password:
       - Go to Google Account settings
       - Security > 2-Step Verification > App passwords
       - Generate app password for 'Mail'
    """
    
    # Get email credentials from Airflow connection
    try:
        conn = BaseHook.get_connection('email_smtp')
        sender_email = conn.login
        password = conn.password
    except Exception as e:
        print(f"Warning: Could not get email connection: {e}")
        print("Email notification skipped. Please configure 'email_smtp' connection in Airflow.")
        return
    
    # Configuration
    receiver_email = "nidhi.mallik2001@gmail.com"  # CHANGE THIS
    dag_id = kwargs['dag'].dag_id
    
    # Email content
    subject = f"✅ Airflow Success: {dag_id}"
    body = f"""
    Hi Team,
    
    The ML pipeline in DAG '{dag_id}' has completed successfully!
    
    Tasks completed:
    - Data loading
    - Data preprocessing
    - Model training
    - Model saved
    
    Best regards,
    Airflow Automation
    """
    
    # Create email
    message = MIMEMultipart()
    message['Subject'] = subject
    message['From'] = sender_email
    message['To'] = receiver_email
    message.attach(MIMEText(body, 'plain'))
    
    # Send email
    try:
        print(f"Sending email to {receiver_email}...")
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        
        print(f"✅ Success email sent to {receiver_email}")
        
    except Exception as e:
        print(f"❌ Error sending email: {e}")
        raise