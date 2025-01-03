# Research & RAG Assistant

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
    - [Company Research Tab](#1-company-research-tab-)
    - [Document Q&A Tab](#2-document-q-a-tab-)
6. [Workflow](#workflow)
7. [Demo Link](#demo-link)
8. [Contributing](#contributing)
9. [License](#license)

---

## Introduction

The Research & RAG Assistant is an AI-powered web application designed to assist users in two main areas:
1. **Company Research**: Automatically generates detailed research reports about companies, including their industry trends and AI implementation opportunities.
2. **Document Q&A**: Enables users to upload documents and have intelligent conversations about their content, combining document knowledge with web search capabilities for comprehensive answers.

This application is ideal for businesses, researchers, and professionals seeking efficient tools for company analysis and document-based inquiry.

---

## Prerequisites

To use the Research & RAG Assistant, you will need:
1. **Python**: Version 3.10 or later.
2. **API Keys**:
   - OpenAI API Key: Required for AI-powered functionalities.
   - Tavily API Key: Required for web search in the Company Research module.
   - Pinecone API Key: Required for managing the document storage system.
3. **Supported File Formats**: For Document Q&A, ensure your files are in one of the supported formats (PDF, Word, TXT, PowerPoint, Markdown).

---

## Installation

Follow these steps to set up the Research & RAG Assistant on your local machine:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/research-rag-assistant.git
   cd research-rag-assistant
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

---

## Configuration

To configure the application, follow these steps:

### API Keys
1. Navigate to the **Configuration** section in the app.
2. Enter your API keys:
   - OpenAI API Key (required for both features)
   - Tavily API Key (required for Company Research)
   - Pinecone API Key (required for Document Q&A)
3. Click the **Save** button. A green success message will confirm that the configuration is complete.

### Optional Settings
- **Web Search**: For Document Q&A, you can enable web search to include online resources in the response.

---

## Usage

### 1. Company Research Tab (🔍)
1. **Setup**:
   - Enter your OpenAI and Tavily API keys in the Configuration section.
2. **Generate Report**:
   - Enter the company name (e.g., "Tesla") and industry (e.g., "Automotive").
   - Click **Generate Research Report**.
3. **Download Report**:
   - The system creates a comprehensive PDF report and provides a download button.

### 2. Document Q&A Tab (💬)
1. **Setup**:
   - Enter your OpenAI and Pinecone API keys in the Configuration section.
   - Optionally enable web search for additional context.
2. **Upload Documents**:
   - Select and upload your documents.
   - Click **Process Documents** to analyze them.
3. **Start Chatting**:
   - Use the chat interface to ask questions about your documents.
   - View responses, which combine document knowledge and optional web search results.

---

## Workflow

The following diagram illustrates the workflow of the Research & RAG Assistant:

![Workflow Diagram](https://via.placeholder.com/800x400.png?text=Workflow+Diagram)

### Step-by-Step Workflow

1. **API Key Configuration**:
   - User enters OpenAI, Tavily, and Pinecone API keys in the app configuration section.

2. **Company Research Process**:
   - User provides a company name and industry.
   - The system performs web searches, analyzes data, and generates a detailed PDF report.

3. **Document Q&A Process**:
   - User uploads documents in supported formats.
   - The system processes and stores document data in Pinecone.
   - User interacts with the chat interface to ask questions.
   - The system retrieves relevant document content and optionally uses web search for additional context.

4. **Output Delivery**:
   - PDF reports for company research are available for download.
   - Chat responses are displayed in real-time, combining document and web knowledge.

---

## Demo Link

Try the live demo here: [Research & RAG Assistant](https://example-link-to-demo.com)

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

