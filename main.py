import os
import re
import json
import google.generativeai as genai
import numpy as np
import customtkinter as ctk
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
from typing import List, Dict, Optional
import textwrap

# 🔑 Load our secret ingredients (environment variables)
load_dotenv()

# 🧹 --- Text Cleaning & Processing Magic ---

class DataPreprocessor:
    def __init__(self):
        self.chunk_size = 512
        self.overlap = 50

    def extract_text_from_pdf(self, file_path):
        try:
            text = ""
            with open(file_path, "rb") as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"  # Add newline between pages
            return text
        except Exception as e:
            print(f"Error reading PDF: {str(e)}")
            return ""

    def extract_text_from_docx(self, file_path):
        try:
            doc = Document(file_path)
            text = " ".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            print(f"Error reading DOCX: {str(e)}")
            return ""

    def extract_text_from_txt(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
            return text
        except Exception as e:
            print(f"Error reading TXT: {str(e)}")
            return ""

    def clean_text(self, text):
        """Enhanced text cleaning with better preservation of meaningful content"""
        if not text:
            return ""

        # 1. Replace multiple newlines with single newline
        text = re.sub(r'\n+', '\n', text)

        # 2. Remove special characters but preserve important punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\'\"()\n-]', '', text)

        # 3. Remove extra whitespace while preserving paragraph structure
        text = '\n'.join(line.strip() for line in text.split('\n'))

        return text.strip()

    def chunk_text(self, text, chunk_size=None, overlap=None):
        """Improved text chunking with sentence awareness"""
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap

        if not text:
            return []

        # Split into sentences first using a raw string for the regex pattern
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_length = len(sentence_words)

            if current_length + sentence_length <= chunk_size:
                current_chunk.extend(sentence_words)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = sentence_words
                current_length = sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

# 🧠 --- Making Text Smart (Embedding Generation & Storage) ---

class EmbeddingManager:
    def __init__(self, embeddings_file="embeddings.json"):
        self.embedding_model_name = "models/text-embedding-004"
        self.embeddings_file = embeddings_file
        self.embeddings_data = self.load_embeddings()

    def generate_embeddings(self, text_chunks):
        """
        Generates embeddings for a list of text chunks.
        Returns the embedding vector directly without the wrapper.
        """
        if not isinstance(text_chunks, list):
            text_chunks = [text_chunks]

        response = genai.embed_content(
            model=self.embedding_model_name,
            content=text_chunks,
            task_type="semantic_similarity",
        )

        # Extract just the embedding vector
        embeddings = response['embedding']
        if len(text_chunks) == 1:
            return embeddings  # Return single embedding for single text
        return embeddings

    def save_embeddings(self, text_chunks, embeddings):
        new_data = []
        for chunk, embedding in zip(text_chunks, embeddings):
            new_data.append({"text": chunk, "embedding": embedding})
        self.embeddings_data.extend(new_data)
        with open(self.embeddings_file, "w") as f:
            json.dump(self.embeddings_data, f)

    def load_embeddings(self):
        try:
            with open(self.embeddings_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def find_similar_chunks(self, query_embedding, top_k=3):
        """
        Find similar chunks using cosine similarity.
        Handles both single and batch query embeddings.
        """
        if not self.embeddings_data:
            return []

        similarities = []
        query_embedding = np.array(query_embedding).flatten()  # Ensure 1D array

        for data in self.embeddings_data:
            embedding = np.array(data["embedding"]).flatten()  # Ensure 1D array
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append(similarity)

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.embeddings_data[i]["text"] for i in top_indices]

# 🤖 --- The Brain Behind the Chat ---

class ChatModel:
    def __init__(self, api_key):
        self.generation_config = {
            "temperature": 0.7,  # Increased for more creative responses while maintaining coherence
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        # Configure the API key for the generative AI model
        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-8b",
            generation_config=self.generation_config
        )
        self.chat_session = None

    def start_chat(self, system_prompt):
        """
        Initialize chat with the system prompt
        """
        try:
            self.chat_session = self.model.start_chat()
            self.chat_session.send_message(system_prompt)
            return True
        except Exception as e:
            print(f"Error starting chat: {str(e)}")
            return False

    def generate_response(self, user_query, context_chunks):
        try:
            # Create a more natural prompt combining context and character
            context_text = "\n".join(context_chunks) if context_chunks else ""

            prompt = (
                f"Using the following context information (but don't mention it directly):\n"
                f"{context_text}\n\n"
                f"Respond to: {user_query}\n\n"
                f"Response:"
            )

            response = self.chat_session.send_message(prompt)
            return response.text
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"

# 🎨 --- The Pretty Face (Modern Chat UI) ---

class ModernChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 🪟 Setting up our cozy chat window
        self.title("Shadow wizard kitten gang")
        self.geometry("1200x800")

        # 📐 Making everything fit just right
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        ctk.set_appearance_mode("dark")

        # 🎨 Our lovely color palette
        self.colors = {
            "primary": "#2563eb",
            "secondary": "#374151",  # Gray-700 for dark mode
            "accent": "#22c55e",
            "text": "#ffffff",  # White text for dark mode
            "border": "#4b5563",  # Gray-600 for borders
        }

        # 🔧 Setting up the behind-the-scenes magic
        self.preprocessor = DataPreprocessor()
        self.embedding_manager = EmbeddingManager()
        self.chat_model = None

        # ✨ Creating our beautiful UI elements
        self.create_header()
        self.create_sidebar()
        self.create_chat_area()

        # 🎛️ Setting initial UI states (everything's disabled until API key is set)
        self.update_ui_state(api_key_set=False)

    def create_header(self):
        self.header_frame = ctk.CTkFrame(self, fg_color=self.colors["secondary"], corner_radius=0)
        self.header_frame.grid(row=0, column=0, columnspan=2, sticky="ew")

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Ask away kitten...",
            font=("Arial Bold", 24),
            text_color=self.colors["text"]
        )
        self.title_label.pack(pady=20)

    def create_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, fg_color=self.colors["secondary"])
        self.sidebar_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        self.sidebar_frame.grid_rowconfigure(3, weight=1)

        self.create_api_section()
        self.create_upload_section()
        self.create_prompt_section()

    def create_api_section(self):
        api_card = ctk.CTkFrame(self.sidebar_frame, fg_color=self.colors["secondary"])
        api_card.pack(fill="x", pady=(0, 20), padx=10)

        api_title = ctk.CTkLabel(
            api_card,
            text="API Configuration",
            font=("Arial Bold", 16),
            text_color=self.colors["text"]
        )
        api_title.pack(pady=10, padx=10)

        self.api_key_entry = ctk.CTkEntry(
            api_card,
            placeholder_text="Enter Gemini API Key",
            width=280
        )
        self.api_key_entry.pack(pady=(0, 10), padx=10)

        self.api_key_button = ctk.CTkButton(
            api_card,
            text="Set API Key",
            command=self.set_api_key,
            fg_color=self.colors["primary"],
            hover_color="#1d4ed8"
        )
        self.api_key_button.pack(pady=(0, 10), padx=10)

    def create_upload_section(self):
        upload_card = ctk.CTkFrame(self.sidebar_frame, fg_color=self.colors["secondary"])
        upload_card.pack(fill="x", pady=(0, 20), padx=10)

        upload_title = ctk.CTkLabel(
            upload_card,
            text="Knowledge Base",
            font=("Arial Bold", 16),
            text_color=self.colors["text"]
        )
        upload_title.pack(pady=10, padx=10)

        self.upload_button = ctk.CTkButton(
            upload_card,
            text="Upload Document",
            command=self.upload_file,
            fg_color=self.colors["primary"],
            text_color=self.colors["text"],
            hover_color="#1d4ed8"
        )
        self.upload_button.pack(pady=(0, 10), padx=10)

    def create_prompt_section(self):
        prompt_card = ctk.CTkFrame(self.sidebar_frame, fg_color=self.colors["secondary"])
        prompt_card.pack(fill="x", pady=(0, 20), padx=10)

        prompt_title = ctk.CTkLabel(
            prompt_card,
            text="Chat Configuration",
            font=("Arial Bold", 16),
            text_color=self.colors["text"]
        )
        prompt_title.pack(pady=10, padx=10)

        self.prompt_entry = ctk.CTkTextbox(
            prompt_card,
            height=100,
            width=280,
            fg_color="white",  # Light chatbox
            text_color="#000000"  # Black text for chatbox
        )
        self.prompt_entry.pack(pady=(0, 10), padx=10)
        self.prompt_entry.insert("1.0", "You are a helpful and friendly AI assistant.")

        self.start_chat_button = ctk.CTkButton(
            prompt_card,
            text="Start Chat",
            command=self.start_chat,
            fg_color=self.colors["accent"],
            hover_color="#16a34a"
        )
        self.start_chat_button.pack(pady=(0, 10), padx=10)

    def create_chat_area(self):
        self.chat_frame = ctk.CTkFrame(self, fg_color=self.colors["secondary"])
        self.chat_frame.grid(row=1, column=1, sticky="nsew", padx=20, pady=20)
        self.chat_frame.grid_rowconfigure(0, weight=1)
        self.chat_frame.grid_columnconfigure(0, weight=1)

        self.chat_output = ctk.CTkTextbox(
            self.chat_frame,
            wrap="word",
            font=("Arial", 14),
            text_color="#000000",
            fg_color="white",  # Light chatbox background
            corner_radius=10
        )
        self.chat_output.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 0))

        input_frame = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        input_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        input_frame.grid_columnconfigure(0, weight=1)

        self.chat_input = ctk.CTkEntry(
            input_frame,
            placeholder_text="Type your message...",
            height=40,
            text_color="#000000",
            fg_color="white"  # Light input field background
        )
        self.chat_input.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.chat_input.bind("<Return>", self.send_message)

        self.send_button = ctk.CTkButton(
            input_frame,
            text="Send",
            width=100,
            height=40,
            command=self.send_message,
            fg_color=self.colors["primary"],
            hover_color="#1d4ed8"
        )
        self.send_button.grid(row=0, column=1)

    def update_ui_state(self, api_key_set: bool):
        state = "normal" if api_key_set else "disabled"
        self.upload_button.configure(state=state)
        self.start_chat_button.configure(state=state)
        self.chat_input.configure(state=state)
        self.send_button.configure(state=state)

    def set_api_key(self):
        api_key = self.api_key_entry.get()
        if api_key:
            self.chat_model = ChatModel(api_key)
            self.update_ui_state(api_key_set=True)
            self.display_message("API Key set successfully. You can now start the chat.")
        else:
            self.display_message("Please enter a valid API Key.")

    def upload_file(self):
        if not self.chat_model:
            self.display_message("Please set the API Key first.")
            return

        file_path = ctk.filedialog.askopenfilename(
            filetypes=[
                ("PDF Files", "*.pdf"),
                ("DOCX Files", "*.docx"),
                ("TXT Files", "*.txt"),
            ]
        )

        if file_path:
            # 📚 Let's see what kind of document we're dealing with
            if file_path.endswith(".pdf"):
                text = self.preprocessor.extract_text_from_pdf(file_path)
            elif file_path.endswith(".docx"):
                text = self.preprocessor.extract_text_from_docx(file_path)
            elif file_path.endswith(".txt"):
                text = self.preprocessor.extract_text_from_txt(file_path)
            else:
                self.display_message("Unsupported file format.")
                return

            # 🧮 Time to make this text computer-friendly!
            cleaned_text = self.preprocessor.clean_text(text)
            text_chunks = self.preprocessor.chunk_text(cleaned_text)
            embeddings = self.embedding_manager.generate_embeddings(text_chunks)
            self.embedding_manager.save_embeddings(text_chunks, embeddings)

            self.display_message(f"File processed successfully: {file_path}")

    def start_chat(self):
        if not self.chat_model:
            self.display_message("Please set the API Key first.")
            return

        system_prompt = self.prompt_entry.get("1.0", "end-1c")
        success = self.chat_model.start_chat(system_prompt)

        if success:
            self.display_message("Chat started successfully!")
        else:
            self.display_message("Failed to start chat. Please try again.")

    def send_message(self, event=None):
        if not self.chat_model:
            self.display_message("Please set the API Key first.")
            return

        user_message = self.chat_input.get()
        if not user_message.strip():
            return

        # 🧹 Clear the input field for the next message
        self.chat_input.delete(0, "end")

        # 💬 Show what the user said
        self.display_message(f"You: {user_message}", is_user=True)

        # 🤔 Time to think and generate a smart response
        query_embedding = self.embedding_manager.generate_embeddings([user_message])[0]
        context_chunks = self.embedding_manager.find_similar_chunks(query_embedding)
        response = self.chat_model.generate_response(user_message, context_chunks)

        # 🤖 Show what our AI friend has to say
        self.display_message(f"AI: {response}")

    def display_message(self, message: str, is_user: bool = False):
        self.chat_output.configure(state="normal")
        if self.chat_output.get("1.0", "end-1c"):
            self.chat_output.insert("end", "\n\n")

        wrapped_message = textwrap.fill(message, width=80)
        tag_name = "user" if is_user else "ai"
        self.chat_output.insert("end", wrapped_message, tag_name)

        if is_user:
            self.chat_output.tag_config(tag_name, foreground=self.colors["primary"])
        else:
            self.chat_output.tag_config(tag_name, foreground="#000000")

        self.chat_output.configure(state="disabled")
        self.chat_output.see("end")

if __name__ == "__main__":
    app = ModernChatApp()
    app.mainloop()