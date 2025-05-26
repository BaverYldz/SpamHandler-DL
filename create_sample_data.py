import pandas as pd
import os

def create_sample_data():
    """
    Create a sample spam/ham dataset if real dataset doesn't exist or has issues
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Sample data with clear spam and non-spam examples
    data = {
        'class': [
            # Ham (0)
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            # Spam (1)
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        ],
        'text': [
            # Ham examples
            "Hey, how are you doing? Let's meet for coffee tomorrow.",
            "Don't forget to pick up milk on your way home.",
            "The meeting is scheduled for 3pm in the conference room.",
            "I'll be there in 10 minutes, just running a bit late.",
            "Thanks for sending the report, I'll review it today.",
            "Can you please call me when you get a chance?",
            "I'm going to the gym after work, want to join?",
            "Just checked the weather, it's going to rain tomorrow.",
            "Remember to bring your laptop for the presentation.",
            "Hi mom, I'll be home for dinner around 6pm.",
            
            # Spam examples
            "URGENT: You've WON a £1000 cash prize! Call now to claim!",
            "Congratulations! You've been selected for a FREE iPhone! Click here to claim now!",
            "Your account will be suspended! Verify your details now: www.suspicious-link.com",
            "Make $5000 a week working from home! Limited time offer!",
            "FINAL NOTICE: Your payment is overdue. Call immediately to avoid legal action!",
            "FREE entry into our £250 weekly competition! Text WIN to 88088 now!",
            "Hot singles in your area waiting to meet you! Click here for FREE membership!",
            "Investment opportunity! 500% returns guaranteed! Act now before it's too late!",
            "Your computer has a virus! Call our tech support immediately: 1-800-SCAM-NOW",
            "Congratulations! You've won a luxury cruise vacation! Call now to claim!"
        ]
    }
    
    df = pd.DataFrame(data)
    output_path = os.path.join(data_dir, 'twitter_spam_data.csv')
    
    if os.path.exists(output_path):
        print(f"Sample data file already exists at {output_path}")
        overwrite = input("Do you want to overwrite it? (y/n): ")
        if overwrite.lower() != 'y':
            print("Operation cancelled.")
            return
    
    df.to_csv(output_path, index=False)
    print(f"Sample data created at {output_path}")
    print("Now you can run the preprocessing and model training:")
    print("1. python src/preprocessing.py")
    print("2. python src/main.py")

if __name__ == "__main__":
    create_sample_data()
