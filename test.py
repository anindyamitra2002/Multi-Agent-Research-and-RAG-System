from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

def generate_pdf_report(state, llm):
    """Generate final PDF report with all collected information."""
    research_data = state['research_results']
    use_cases = state['use_cases']
    resources = state['resources']
    search_queries = state['search_queries']
    company = state['company']
    industry = state['industry']
    
    # Format resources for the prompt
    datasets_text = "\n".join([
        f"- **[{d['platform']}] {d['title']}**\n  - Description: {d['description']}\n  - Query: {d['query']}\n  - URL: {d['url']}"
        for d in resources['datasets']
    ])
    
    implementations_text = "\n".join([
        f"- **[{d['platform']}] {d['title']}**\n  - Description: {d['description']}\n  - Query: {d['query']}\n  - URL: {d['url']}"
        for d in resources['implementations']
    ])
    
    prompt = PromptTemplate.from_template(
        """Create a comprehensive GenAI & ML implementation proposal for {company} in the {industry} industry.
        
        # GenAI & ML Implementation Proposal for {company}
        
        ## Executive Summary
        [Generate a concise yet comprehensive executive summary that outlines:
        - Company's current position in the {industry} industry
        - Key opportunities for AI/ML implementation
        - Expected business impact and ROI
        - Implementation timeline overview]
        
        ## Industry and Company Analysis
        {analysis}
        
        ## Strategic AI/ML Implementation Opportunities
        
        Based on the industry analysis and company strengths, here are the key AI/ML implementation opportunities:
        
        [For each use case, provide:
        
        **Use Case i: [Name of Use Case]**
         **Objective/Use Case**: [Clear description of the problem being solved]
         **AI Application**: [Specific AI/ML technologies and implementation approach]
         **Cross-Functional Benefits**:
          - **[Department 1]**: [Specific benefits]
          - **[Department 2]**: [Specific benefits]
          - **[Department 3]**: [Specific benefits]
         **Implementation Requirements**:
          - Technical Infrastructure
          - Data Requirements
          - Timeline
          - Resource Needs
         **Expected ROI**:
           - Quantitative Benefits
           - Qualitative Benefits
        ]
        
        {use_cases}
        
        ## Technical Implementation Framework
        
        ### Available Resources
        
        #### Recommended Datasets
        {datasets}
        
        #### Development Resources & Tools
        {implementations}
        
        ## Implementation Roadmap
        
        ### Phase 1: Foundation (Months 1-3)
        [Detail specific steps, milestones, and deliverables]
        
        ### Phase 2: Development (Months 4-6)
        [Detail specific steps, milestones, and deliverables]
        
        ### Phase 3: Deployment (Months 7-9)
        [Detail specific steps, milestones, and deliverables]
        
        ## Risk Assessment and Mitigation
        [Identify key risks and mitigation strategies]
        
        ## Next Steps and Recommendations
        [Provide clear, actionable next steps with timelines and ownership]
        
        ## References and Resources
        [List all technical references, datasets, and implementation resources]
        
        Format the content professionally with clear sections and detailed subsections."""
    )
    
    chain = prompt | llm | StrOutputParser()
    markdown_content = chain.invoke({
        "company": company,
        "industry": industry,
        "analysis": research_data['analysis'],
        "use_cases": use_cases,
        "datasets": datasets_text,
        "implementations": implementations_text
    })
    
    # Convert markdown to PDF
    import tempfile
    import os
    import markdown2
    from xhtml2pdf import pisa
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    pdf_path = f"{company}_AI_Implementation_Proposal.pdf"
    
    # Convert markdown to HTML
    html_content = markdown2.markdown(markdown_content, extras=['tables', 'break-on-newline'])
    
    # HTML template with enhanced styles
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{
                size: A4;
                margin: 2.5cm;
                @frame footer {{
                    -pdf-frame-content: footerContent;
                    bottom: 1cm;
                    margin-left: 1cm;
                    margin-right: 1cm;
                    height: 1cm;
                }}
            }}
            body {{
                font-family: Helvetica, Arial, sans-serif;
                font-size: 11pt;
                line-height: 1.6;
                color: #2c3e50;
            }}
            h1 {{
                font-size: 24pt;
                color: #1a237e;  /* Deep Blue */
                text-align: center;
                margin-bottom: 2cm;
                font-weight: bold;
            }}
            h2 {{
                font-size: 18pt;
                color: #283593;  /* Slightly lighter blue */
                margin-top: 1.5cm;
                border-bottom: 2px solid #3949ab;
                padding-bottom: 0.3cm;
            }}
            h3 {{
                font-size: 14pt;
                color: #3949ab;  /* Medium blue */
                margin-top: 1cm;
            }}
            h4 {{
                font-size: 12pt;
                color: #5c6bc0;  /* Light blue */
                margin-top: 0.8cm;
            }}
            p {{
                text-align: justify;
                margin-bottom: 0.5cm;
            }}
            ul {{
                margin-left: 0;
                padding-left: 1cm;
                margin-bottom: 0.5cm;
            }}
            li {{
                margin-bottom: 0.3cm;
            }}
            a {{
                color: #3f51b5;
                text-decoration: none;
            }}
            strong {{
                color: #283593;
            }}
            .use-case {{
                background-color: #f5f7fa;
                padding: 1cm;
                margin: 0.5cm 0;
                border-left: 4px solid #3949ab;
            }}
            .benefit {{
                margin-left: 1cm;
                color: #34495e;
            }}
        </style>
    </head>
    <body>
        {html_content}
        <div id="footerContent" style="text-align: center; font-size: 8pt; color: #7f8c8d;">
            Page <pdf:pagenumber> of <pdf:pagecount>
        </div>
    </body>
    </html>
    """
    
    # Convert HTML to PDF
    with open(pdf_path, "w+b") as pdf_file:
        pisa.CreatePDF(
            html_template,
            dest=pdf_file
        )
    
    return {"final_report": pdf_path}

dummy_state = {
    "research_results": {
        "analysis": """
            The {industry} industry is undergoing rapid transformation with AI/ML technologies. 
            {company} has a significant opportunity to lead innovation by leveraging data-driven insights 
            and automating key processes.
        """
    },
    "use_cases": """
        1. Predictive maintenance to reduce equipment downtime.
        2. Customer segmentation and personalized marketing strategies.
        3. Supply chain optimization using AI for demand forecasting.
    """,
    "resources": {
        "datasets": [
            {
                "platform": "Kaggle",
                "title": "Customer Segmentation Dataset",
                "description": "A dataset containing customer demographics and purchase history.",
                "query": "customer segmentation dataset AI",
                "url": "https://www.kaggle.com/datasets/customer-segmentation"
            },
            {
                "platform": "UCI Machine Learning Repository",
                "title": "Demand Forecasting Dataset",
                "description": "Data on product sales over time for demand forecasting.",
                "query": "demand forecasting dataset",
                "url": "https://archive.ics.uci.edu/ml/datasets/Demand+Forecasting"
            }
        ],
        "implementations": [
            {
                "platform": "GitHub",
                "title": "Predictive Maintenance Model",
                "description": "A repository with code for implementing predictive maintenance using AI.",
                "query": "predictive maintenance GitHub",
                "url": "https://github.com/username/predictive-maintenance"
            },
            {
                "platform": "Medium",
                "title": "AI in Supply Chain Optimization",
                "description": "An article with examples of AI applications in supply chain optimization.",
                "query": "AI supply chain optimization",
                "url": "https://medium.com/ai-supply-chain-optimization"
            }
        ]
    },
    "search_queries": {
        "dataset_queries": [
            "Find datasets for customer segmentation.",
            "Locate datasets for predictive maintenance.",
            "Discover datasets for demand forecasting."
        ],
        "implementation_queries": [
            "Search for AI implementation examples in supply chain optimization.",
            "Look for predictive maintenance implementation code.",
            "Find case studies on personalized marketing strategies using AI."
        ]
    },
    "company": "Tech Innovators Ltd.",
    "industry": "Manufacturing"
}

llm = ChatOpenAI(
        model="gpt-3.5-turbo-0125",
        temperature=0.3,
        timeout=None,
        max_retries=2)

# Run the function
pdf_report = generate_pdf_report(dummy_state, llm)

# Print the PDF path
print("PDF report generated at:", pdf_report["final_report"])
