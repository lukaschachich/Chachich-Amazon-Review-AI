import re # Importing the regular expressions module, it is used for text preprocessing

# Function to preprocess text data

def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
      
    text = re.sub(r"<.*?>", "", text) # Remove HTML tags

    text = re.sub(r"http\S+|www\S+", "", text) # Remove links from text

    return text.strip()  # Strip remaining whitespace around text

print(preprocess_text("  This is a <b>sample</b> text with a link: https://example.com  "))  # Example usage