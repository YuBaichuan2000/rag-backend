# cli_test.py
import argparse
import requests
import json
import sys
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown

# Load environment variables from .env
load_dotenv()

# Initialize console for pretty printing
console = Console()

class ChatbotTester:
    """Simple CLI for testing RAG Chatbot API"""
    
    def __init__(self, host, user_id):
        self.host = host
        self.user_id = user_id
        self.thread_id = None
    
    def display_welcome(self):
        """Display welcome message"""
        console.print(Panel(
            "[bold blue]RAG Chatbot API Tester[/bold blue]\n\n"
            "Type your message to chat\n"
            "Commands:\n"
            "  /new - Start a new conversation\n"
            "  /url <url> - Upload a document from URL\n"
            "  /file <path> - Upload a local file\n"
            "  /list - List conversations\n"
            "  /exit - Exit the tester",
            title="Welcome",
            expand=False
        ))
    
    def chat(self, message):
        """Send a chat message to the API"""
        try:
            endpoint = f"{self.host}/chat"
            payload = {
                "message": message,
                "user_id": self.user_id
            }
            
            # Add thread_id if we have one
            if self.thread_id:
                payload["thread_id"] = self.thread_id
            
            # Send request
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            self.thread_id = data["thread_id"]
            
            # Display response
            console.print("\n[bold green]AI:[/bold green]")
            console.print(Markdown(data["response"]))
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
    
    def new_conversation(self):
        """Start a new conversation"""
        try:
            endpoint = f"{self.host}/new-conversation?user_id={self.user_id}"
            response = requests.post(endpoint)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            self.thread_id = data["thread_id"]
            
            # Display response
            console.print(f"\n[bold green]New conversation started![/bold green]")
            console.print(f"Thread ID: {self.thread_id}")
            console.print(Markdown(data["response"]))
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
    
    def upload_url(self, url, title=None):
        """Upload document from URL"""
        try:
            endpoint = f"{self.host}/upload-url"
            payload = {
                "url": url,
                "user_id": self.user_id
            }
            
            if title:
                payload["title"] = title
            
            # Send request
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Display response
            console.print(f"\n[bold green]Document uploaded successfully![/bold green]")
            console.print(f"Document ID: {data['document_id']}")
            console.print(f"Title: {data['title']}")
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
    
    # cli_test.py (continued)
    def upload_file(self, file_path):
        """Upload document from file"""
        try:
            if not os.path.exists(file_path):
                console.print(f"[bold red]Error:[/bold red] File {file_path} does not exist")
                return
            
            filename = os.path.basename(file_path)
            endpoint = f"{self.host}/upload-file"
            
            # Determine content type based on file extension
            file_extension = os.path.splitext(filename)[1].lower()
            content_type = None
            if file_extension == '.pdf':
                content_type = 'application/pdf'
            elif file_extension == '.txt':
                content_type = 'text/plain'
            else:
                console.print(f"[bold red]Error:[/bold red] Unsupported file type {file_extension}")
                return
            
            # Use context manager to properly handle file
            with open(file_path, 'rb') as file_obj:
                files = {'file': (filename, file_obj, content_type)}
                data = {'user_id': self.user_id}
                
                # Send request
                response = requests.post(endpoint, files=files, data=data)
                response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            # Display response
            console.print(f"\n[bold green]File uploaded successfully![/bold green]")
            console.print(f"Document ID: {response_data['document_id']}")
            console.print(f"Title: {response_data['title']}")
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
    
    def list_conversations(self):
        """List all conversations"""
        try:
            endpoint = f"{self.host}/conversations?user_id={self.user_id}"
            response = requests.get(endpoint)
            response.raise_for_status()
         
            
            # Parse response
            data = response.json()
            
            # Display response
            if not data['conversations']:
                console.print("\n[bold yellow]No conversations found[/bold yellow]")
                return
            
            console.print("\n[bold green]Conversations:[/bold green]")
            for idx, conv in enumerate(data['conversations'], 1):
                console.print(f"{idx}. Thread ID: {conv['thread_id']}")
                if 'last_active' in conv:
                    console.print(f"   Last active: {conv['last_active']}")
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
    
    def run(self):
        """Run the CLI tester"""
        self.display_welcome()
        
        # Start a new conversation by default
        self.new_conversation()
        
        while True:
            # Get user input
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
            
            # Check for commands
            if user_input.startswith('/'):
                command_parts = user_input.split()
                command = command_parts[0].lower()
                
                if command == '/exit':
                    console.print("[bold yellow]Exiting...[/bold yellow]")
                    break
                elif command == '/new':
                    self.new_conversation()
                elif command == '/url':
                    if len(command_parts) < 2:
                        console.print("[bold red]Error:[/bold red] URL required")
                        continue
                    url = command_parts[1]
                    title = None
                    if len(command_parts) > 2:
                        title = ' '.join(command_parts[2:])
                    self.upload_url(url, title)
                elif command == '/file':
                    if len(command_parts) < 2:
                        console.print("[bold red]Error:[/bold red] File path required")
                        continue
                    file_path = command_parts[1]
                    self.upload_file(file_path)
                elif command == '/list':
                    self.list_conversations()
                else:
                    console.print(f"[bold red]Unknown command:[/bold red] {command}")
            else:
                # Regular chat message
                self.chat(user_input)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RAG Chatbot API CLI Tester")
    parser.add_argument("--host", default=os.getenv("API_URL", "http://localhost:8001"), 
                        help="API host URL")
    parser.add_argument("--user", default=os.getenv("TEST_USER_ID", "test-user-123"), 
                        help="User ID for testing")
    args = parser.parse_args()
    
    # Initialize and run tester
    tester = ChatbotTester(
        host=args.host,
        user_id=args.user
    )
    tester.run()