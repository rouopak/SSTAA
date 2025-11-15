import speech_recognition as sr
import os
import webbrowser
import datetime
import platform
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import time
import threading

# https://youtu.be/Z3ZAJoi4x6Q

# Initialize TTS (using edge-tts which is easier to install)
tts_available = False
try:
    import edge_tts
    tts_available = True
    print("Edge TTS initialized successfully")
except ImportError:
    try:
        from gtts import gTTS
        tts_available = True
        print("gTTS initialized successfully")
    except ImportError:
        print("Warning: No TTS library available. Install edge-tts or gtts for text-to-speech")
        print("Run: pip install edge-tts")

# Initialize Llama model from HuggingFace
llama_model = None
llama_tokenizer = None
try:
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Using Llama 2, can be changed to other Llama models
    print("Loading Llama model from HuggingFace...")
    llama_tokenizer = AutoTokenizer.from_pretrained(model_name)
    llama_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    print("Llama model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load Llama model: {e}")
    print("AI features may be limited. Trying alternative model...")
    try:
        # Fallback to a smaller model
        model_name = "microsoft/DialoGPT-medium"
        llama_tokenizer = AutoTokenizer.from_pretrained(model_name)
        llama_model = AutoModelForCausalLM.from_pretrained(model_name)
        print("Fallback model loaded successfully")
    except Exception as e2:
        print(f"Could not load fallback model: {e2}")

def say(text):
    """Use TTS to speak text"""
    try:
        if not tts_available:
            print(text)  # Fallback to printing text
            return
        
        temp_file = "temp_tts_output.mp3"
        
        # Try edge-tts first (preferred, works offline)
        try:
            import edge_tts
            import asyncio
            
            async def generate_speech():
                communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
                await communicate.save(temp_file)
            
            # Run async function
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            loop.run_until_complete(generate_speech())
            
        except (ImportError, Exception) as e:
            # Fallback to gTTS
            try:
                from gtts import gTTS
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(temp_file)
            except Exception as e2:
                print(f"TTS error: {e2}")
                print(text)
                return
        
        # Play the audio file
        if platform.system() == "Windows":
            # Use os.startfile which works natively on Windows
            os.startfile(temp_file)
            # Clean up after delay (non-blocking)
            threading.Timer(5.0, lambda: os.remove(temp_file) if os.path.exists(temp_file) else None).start()
        elif platform.system() == "Darwin":  # macOS
            os.system(f"afplay {temp_file}")
            time.sleep(0.5)
            if os.path.exists(temp_file):
                os.remove(temp_file)
        else:  # Linux
            os.system(f"mpg123 {temp_file} 2>/dev/null || mpg321 {temp_file} 2>/dev/null || aplay {temp_file}")
            time.sleep(0.5)
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
    except Exception as e:
        print(f"Text-to-speech error: {e}")
        print(text)  # Fallback to printing text

def get_ai_response(query):
    """Get AI response using Llama model from HuggingFace"""
    if llama_model is None or llama_tokenizer is None:
        return "I'm sorry, the AI model is not available. Please try again later."
    
    try:
        # Format the prompt for Llama (using chat template if available)
        if hasattr(llama_tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": query}]
            prompt = llama_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"User: {query}\nAssistant:"
        
        # Tokenize input
        inputs = llama_tokenizer.encode(prompt, return_tensors="pt")
        
        # Move inputs to same device as model
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        # Generate response
        with torch.no_grad():
            outputs = llama_model.generate(
                inputs,
                max_new_tokens=150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=llama_tokenizer.eos_token_id if llama_tokenizer.pad_token_id is None else llama_tokenizer.pad_token_id,
                eos_token_id=llama_tokenizer.eos_token_id
            )
        
        # Decode response (only the new tokens)
        response = llama_tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # Clean up response
        response = response.strip()
        
        return response if response else "I'm not sure how to respond to that."
    except Exception as e:
        print(f"AI response error: {e}")
        return "I encountered an error processing your request. Please try again."

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        # r.pause_threshold =  0.6
        audio = r.listen(source)
        try:
            print("Recognizing...")
            query = r.recognize_google(audio, language="en-in")
            print(f"User said: {query}")
            return query
        except Exception as e:
            print(f"Error: {e}")
            return "Some Error Occurred. Sorry from Jarvis"

if __name__ == '__main__':
    print('Welcome to Jarvis A.I')
    say("Jarvis A.I I love RUPAK and Devansh ")
    while True:
        print("Listening...")
        query = takeCommand()
        # todo: Add more sites
        sites = [["youtube", "https://www.youtube.com"], ["wikipedia", "https://www.wikipedia.com"], ["google", "https://www.google.com"], ["Stake", "https://stake.ac"], ["Spotify", "https://www.spotify.com"]]
        for site in sites:
            if f"Open {site[0]}".lower() in query.lower():
                say(f"Opening {site[0]} sir...")
                webbrowser.open(site[1])
        # todo: Add a feature to play a specific song
        if "open music" in query.lower():
            musicPath = os.path.join(os.path.expanduser("~"), "Downloads", "downfall-21371.mp3")
            if platform.system() == "Windows":
                os.startfile(musicPath)
            else:
                os.system(f"open {musicPath}")

        elif "the time" in query.lower():
            hour = datetime.datetime.now().strftime("%H")
            min = datetime.datetime.now().strftime("%M")
            say(f"Sir time is {hour} bajke {min} minutes")

        elif "open facetime".lower() in query.lower():
            if platform.system() == "Windows":
                os.system("start ms-availablenetworks:")
            else:
                os.system(f"open /System/Applications/FaceTime.app")

        elif "open pass".lower() in query.lower():
            if platform.system() == "Windows":
                # Windows equivalent - you may need to adjust the path
                os.system("start passky")
            else:
                os.system(f"open /Applications/Passky.app")

        elif "create file" in query.lower() or "make file" in query.lower():
            # Extract filename from query
            match = re.search(r'(?:create|make)\s+file\s+(?:called|named)?\s*["\']?([^"\']+)["\']?', query.lower())
            if match:
                filename = match.group(1).strip()
                try:
                    with open(filename, 'w') as f:
                        f.write("")  # Create empty file
                    say(f"File {filename} created successfully")
                    print(f"File {filename} created successfully")
                except Exception as e:
                    error_msg = f"Error creating file: {str(e)}"
                    say(error_msg)
                    print(error_msg)
            else:
                say("Please specify a filename. For example: create file test.txt")
        
        elif "delete file" in query.lower() or "remove file" in query.lower():
            # Extract filename from query
            match = re.search(r'(?:delete|remove)\s+file\s+(?:called|named)?\s*["\']?([^"\']+)["\']?', query.lower())
            if match:
                filename = match.group(1).strip()
                try:
                    if os.path.exists(filename):
                        os.remove(filename)
                        say(f"File {filename} deleted successfully")
                        print(f"File {filename} deleted successfully")
                    else:
                        say(f"File {filename} not found")
                        print(f"File {filename} not found")
                except Exception as e:
                    error_msg = f"Error deleting file: {str(e)}"
                    say(error_msg)
                    print(error_msg)
            else:
                say("Please specify a filename. For example: delete file test.txt")

        elif "Jarvis Quit".lower() in query.lower() or "exit" in query.lower():
            say("Goodbye! Shutting down Jarvis.")
            exit()

        elif "reset chat".lower() in query.lower():
            say("Chat history has been reset.")
            print("Chat history reset")

        else:
            # Use AI model for general queries
            print("Processing with AI...")
            ai_response = get_ai_response(query)
            print(f"Jarvis: {ai_response}")
            say(ai_response)





        # say(query)