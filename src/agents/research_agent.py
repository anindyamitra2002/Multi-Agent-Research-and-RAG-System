from langgraph.graph import END, StateGraph, START
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

import re
import asyncio
from typing import TypedDict, List, Optional, Dict
from src.tools.deep_crawler import DeepWebCrawler, ResourceCollectionAgent

class ResearchGraphState(TypedDict):
    company: str
    industry: str
    research_results: Optional[dict]
    use_cases: Optional[str]
    search_queries: Optional[Dict[str, List[str]]]
    resources: Optional[List[dict]]
    final_report: Optional[str]


def clean_text(text):
    """
    Cleans the given text by:
        1. Removing all hyperlinks.
        2. Removing unnecessary parentheses and square brackets.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text with hyperlinks, parentheses, and square brackets removed.
    """
    # Regular expression pattern for matching URLs
    url_pattern = r'https?://\S+|www\.\S+'
    # Remove hyperlinks
    text_without_links = re.sub(url_pattern, '', text)
    
    # Regular expression pattern for matching parentheses and square brackets
    brackets_pattern = r'[\[\]\(\)]'
    # Remove unnecessary brackets
    cleaned_text = re.sub(brackets_pattern, '', text_without_links)
    
    return cleaned_text.strip()


def create_industry_research_workflow(llm):
    async def industry_research(state: ResearchGraphState):
        """Research industry and company using DeepWebCrawler."""
        company = state['company']
        industry = state['industry']
        
        queries = [
            f"{company} company profile services",
        ]
        
        crawler = DeepWebCrawler(
            max_search_results=3,
            max_external_links=1,
            word_count_threshold=100,
            content_filter_type='bm25',
            filter_threshold=0.48
        )
        
        all_results = []
        for query in queries:
            results = await crawler.search_and_crawl(query)
            all_results.extend(results)
        print(all_results)
        combined_content = "\n\n".join([
            f"Title: {clean_text(r['title'])} \n{clean_text(r['content'])}" 
            for r in all_results if r['success']
        ])
        print("Combined Content: ", combined_content)
        prompt = PromptTemplate.from_template(
            """Analyze this research about {company} in the {industry} industry:
            {content}
            
            Provide a comprehensive overview including:
            1. Company Overview
            2. Market Segments
            3. Products and Services
            4. Strategic Focus Areas
            5. Industry Trends
            6. Competitive Position
            
            Format the analysis in clear sections with headers."""
        )
        
        chain = prompt | llm | StrOutputParser()
        analysis = chain.invoke({
            "company": company,
            "industry": industry,
            "content": combined_content
        })
        print("Analysis: ", analysis)
        return {
            "research_results": {
                "analysis": analysis,
                "raw_content": combined_content
            }
        }

    def generate_use_cases_and_queries(state: ResearchGraphState):
        """Generate AI/ML use cases and extract relevant search queries."""
        research_data = state['research_results']
        company = state['company']
        industry = state['industry']
        
        # First generate use cases
        use_case_prompt = PromptTemplate.from_template(
            """Based on this research: 
            
            Analysis: {analysis}
            Raw Research: {raw_content}
            
            Generate innovative use cases where {company} in the {industry} industry can leverage 
            Generative AI and Large Language Models for:
            
            1. Internal Process Improvements
            2. Customer Experience Enhancement
            3. Product/Service Innovation
            4. Data Analytics and Decision Making
            
            For each use case, provide:
            - Clear description
            - Expected benefits
            - Implementation considerations"""
        )
        
        chain = use_case_prompt | llm | StrOutputParser()
        use_cases = chain.invoke({
            "company": company,
            "industry": industry,
            "analysis": research_data['analysis'],
            "raw_content": research_data['raw_content']
        })
        
        # Then extract relevant search queries
        query_extraction_prompt = PromptTemplate.from_template(
            """Based on these AI/ML use cases for {company}:

            {use_cases}
            
            Extract Two specific search queries for finding relevant datasets and implementations.
            
            Provide your response in this exact format:
            DATASET QUERIES:
            - query1
            - query2

            IMPLEMENTATION QUERIES:
            - query1
            - query2

            Make queries specific and technical. Include ML model types, data types, and specific AI techniques."""
        )
        
        chain = query_extraction_prompt | llm | StrOutputParser()
        queries_text = chain.invoke({
            "company": company,
            "use_cases": use_cases
        })
        
        # Parse the text response into structured format
        def parse_queries(text):
            dataset_queries = []
            implementation_queries = []
            current_section = None
            
            for line in text.split('\n'):
                line = line.strip()
                if line == "DATASET QUERIES:":
                    current_section = "dataset"
                elif line == "IMPLEMENTATION QUERIES:":
                    current_section = "implementation"
                elif line.startswith("- "):
                    query = line[2:].strip()
                    if current_section == "dataset":
                        dataset_queries.append(query)
                    elif current_section == "implementation":
                        implementation_queries.append(query)
            
            return {
                "dataset_queries": dataset_queries or ["machine learning datasets business", "AI training data industry"],
                "implementation_queries": implementation_queries or ["AI tools business automation", "machine learning implementation"]
            }

        search_queries = parse_queries(queries_text)
        print("Search_queries: ", search_queries)
        return {
            "use_cases": use_cases,
            "search_queries": search_queries
        }

    async def collect_targeted_resources(state: ResearchGraphState):
        """Find relevant datasets and resources using extracted queries."""
        search_queries = state['search_queries']
        resource_agent = ResourceCollectionAgent(max_results_per_query=5)
        
        # Collect resources using targeted queries
        all_resources = {
            "datasets": [],
            "implementations": []
        }
        
        # Search for datasets
        for query in search_queries['dataset_queries']:
            # Add platform-specific modifiers to queries
            kaggle_query = f"site:kaggle.com/datasets {query}"
            huggingface_query = f"site:huggingface.co/datasets {query}"
            
            resources = await resource_agent.collect_resources()
            
            # Process and categorize results
            if resources.get("kaggle_datasets"):
                all_resources["datasets"].extend([{
                    "title": item["title"],
                    "url": item["url"],
                    "description": item["snippet"],
                    "platform": "Kaggle",
                    "query": query
                } for item in resources["kaggle_datasets"]])
                
            if resources.get("huggingface_datasets"):
                all_resources["datasets"].extend([{
                    "title": item["title"],
                    "url": item["url"],
                    "description": item["snippet"],
                    "platform": "HuggingFace",
                    "query": query
                } for item in resources["huggingface_datasets"]])
        
        # Search for implementations
        for query in search_queries['implementation_queries']:
            github_query = f"site:github.com {query}"
            
            resources = await resource_agent.collect_resources()
            
            if resources.get("github_repositories"):
                all_resources["implementations"].extend([{
                    "title": item["title"],
                    "url": item["url"],
                    "description": item["snippet"],
                    "platform": "GitHub",
                    "query": query
                } for item in resources["github_repositories"]])
        print("Resources: ", all_resources)
        return {"resources": all_resources}

    def generate_pdf_report(state: ResearchGraphState):
        """Generate final PDF report with all collected information."""
        research_data = state['research_results']
        use_cases = state['use_cases']
        resources = state['resources']
        company = state['company']
        industry = state['industry']
        
        # Format resources for manual append later
        datasets_section = "\n## Available Datasets\n"
        if resources.get('datasets'):
            for dataset in resources['datasets']:
                datasets_section += f"  - {dataset['platform']}: {dataset['url']}\n"
        
        implementations_section = "\n## Implementation Resources\n"
        if resources.get('implementations'):
            for impl in resources['implementations']:
                implementations_section += f"  - {impl['platform']}: {impl['url']}\n"
        
        
        prompt = PromptTemplate.from_template(
        """
        # GenAI & ML Implementation Proposal for {company}

        ## Executive Summary
        - **Current Position in the {industry} Industry**: 
        - **Key Opportunities for AI/ML Implementation**: 
        - **Expected Business Impact and ROI**: 
        - **Implementation Timeline Overview**: 

        ## Industry and Company Analysis
        {analysis}

        ## Strategic AI/ML Implementation Opportunities

        Based on the analysis, here are the key opportunities for AI/ML implementation:

        {use_cases}

        Format the report in Markdown for clear sections, headings, and bullet points. Ensure professional formatting with structured subsections.
        """
        )
        
        chain = prompt | llm | StrOutputParser()
        markdown_content = chain.invoke({
            "company": company,
            "industry": industry,
            "analysis": research_data['analysis'],
            "use_cases": use_cases,
        })
        
        if markdown_content.startswith("```markdown") and markdown_content.endswith("```"):
            markdown_content = markdown_content[len("```markdown"):].rstrip("```").strip()
            
        markdown_content += "\n\n" + datasets_section + "\n\n" + implementations_section
        # Convert markdown to PDF
        import tempfile
        import os
        import markdown2
        from xhtml2pdf import pisa
        
        # Create temporary directory and full path for PDF
        temp_dir = tempfile.mkdtemp()
        pdf_filename = f"{company.replace(' ', '_')}_research_report.pdf"
        pdf_path = os.path.join(temp_dir, pdf_filename)
        
        html_content = markdown2.markdown(markdown_content, extras=['tables', 'break-on-newline'])
        # HTML template with enhanced styles (same as before)
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
                    color: #1a237e;
                    text-align: center;
                    margin-bottom: 2cm;
                    font-weight: bold;
                }}
                h2 {{
                    font-size: 18pt;
                    color: #283593;
                    margin-top: 1.5cm;
                    border-bottom: 2px solid #3949ab;
                    padding-bottom: 0.3cm;
                }}
                h3 {{
                    font-size: 14pt;
                    color: #3949ab;
                    margin-top: 1cm;
                }}
                h4 {{
                    font-size: 12pt;
                    color: #5c6bc0;
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
        
        # Convert HTML to PDF with proper error handling
        try:
            with open(pdf_path, "w+b") as pdf_file:
                result = pisa.CreatePDF(
                    html_template,
                    dest=pdf_file
                )
                if result.err:
                    print(f"Error generating PDF: {result.err}")
                    return {"final_report": None}
                
            # Verify the file exists
            if os.path.exists(pdf_path):
                print(f"PDF successfully generated at: {pdf_path}")
                return {"final_report": pdf_path}
            else:
                print("PDF file was not created successfully")
                return {"final_report": None}
                
        except Exception as e:
            print(f"Exception during PDF generation: {str(e)}")
            return {"final_report": None}

    # Create workflow
    workflow = StateGraph(ResearchGraphState)
    
    # Add nodes
    workflow.add_node("industry_research", industry_research)
    workflow.add_node("use_cases_gen", generate_use_cases_and_queries)
    workflow.add_node("resources_gen", collect_targeted_resources)
    workflow.add_node("report", generate_pdf_report)
    
    # Define edges
    workflow.add_edge(START, "industry_research")
    workflow.add_edge("industry_research", "use_cases_gen")
    workflow.add_edge("use_cases_gen", "resources_gen")
    workflow.add_edge("resources_gen", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()

async def run_industry_research(company: str, industry: str, llm):
    """Run the industry research workflow asynchronously."""
    workflow = create_industry_research_workflow(llm)
    
    final_state = None
    output = await workflow.ainvoke(input={
        "company": company,
        "industry": industry
    }, config={"recursion_limit": 5})
    
    return output['final_report']

# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize LLM
        llm = ChatOpenAI(
        model="gpt-3.5-turbo-0125",
        temperature=0.3,
        timeout=None,
        max_retries=2,)
        
        # Run the workflow
        report_path = await run_industry_research(
            company="Adani Defence & Aerospace",
            industry="Defense Engineering and Construction",
            llm=llm
        )
        print(f"Report generated at: {report_path}")
    
    asyncio.run(main())