import nltk
import spacy
import textwrap

nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")

def split_into_sentences(text):
    """Splits text into sentences using NLTK."""
    return nltk.sent_tokenize(text)

def split_into_paragraphs(text, sentence_threshold=3):
    """Splits text into paragraphs by grouping sentences."""
    sentences = split_into_sentences(text)
    paragraphs = []
    temp_paragraph = []

    for sentence in sentences:
        temp_paragraph.append(sentence)
        if len(temp_paragraph) >= sentence_threshold:
            paragraphs.append(" ".join(temp_paragraph))
            temp_paragraph = []

    if temp_paragraph:
        paragraphs.append(" ".join(temp_paragraph))

    return paragraphs

def sliding_window_chunking(text, chunk_size=100, step_size=50):
    """
    Splits text into overlapping chunks using a sliding window.
    This ensures that key information is retained in multiple chunks.
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), step_size):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks

def process_resume_text(text):
    """Applies all three chunking methods to the resume text."""
    sentences = split_into_sentences(text)
    paragraphs = split_into_paragraphs(text)
    sliding_chunks = sliding_window_chunking(text)

    return {
        "sentences": sentences,
        "paragraphs": paragraphs,
        "sliding_chunks": sliding_chunks
    }

# Example usage
resume_text = """
REGIONAL BANKING DISTRICT MANAGER VICE PRESIDENT Professional Summary Motivated tenured manager with demonstrated knowledge of industry best practices and operations. Proven skills in enhancing productivity, efficiency and bottom-line profits with forward-thinking leadership. Bringing over twenty years of experience in financial , customer service, sales , problem solving field, with over 15 years management experience. Ready to take on a challenging, growth-oriented role to provide fulfillment and professional purpose. Skills Negotiation Business planning Staff Management Business Development Strategic planning Financial Management Sales and marketing Leadership Adaptability Customer Service Organization and Time management Creative Enthusiastic Work History Regional Banking District Manager Vice President Company Name City , State June 2011 to Current Responsible for leading, growing , and mantaining customer relationships for local branches wih 3.5 billion in customer deposits Oversaw an average of 125 employees, and diretly managed and led 11 Branch Managers throughout the Westside and Santa Monica area Revitalized operations and realigned plans to better capture new opportunities and take advantage of changes in customer habits. Worked diligently to resolve unique and recurring complaints, promoting loyalty, prioritizing customer needs and enhancing operations . Supervised 11 locations to enforce high-quality standards of operations from loss prevention to operational processes. Conceptualized and implemented strategies to realign operational strategies and enhance personnel management approaches. Met deadlines by proactively managing individual and team tasks and streamlining processes throughtout branches and industry. Spearheaded routine operations and special program initiatives for regional group, including digital implmenation and skill development. Modeled best practices for sales and customer service. Located, developed and promoted talented employees to cultivate a collaborative and hardworking leadership team. Partnering with different lines of business to support client needs and apply unified approach in meeting business expectations and goals Branch Manager Vice President Company Name City , State April 2007 to June 2011 Encouraged employee development and promoted management staff from within. Supervised branch operations and made continuous improvements in each area. Utilized up-to-date information to make effective decisions governing bank operations. Set risk management policies to mitigate bank losses. Defined strategies and made proactive adjustments to maintain results. Provided direction and leadership to all employees . Upheld stringent bank standards for loans, money handling and legal considerations. Represented bank at community events to establish strong ties and promote business. Reviewed historical records, current operational data and forecasting information to identify and capitalize on system enhancement opportunities. Performed banking, business administration and financial tasks to guarantee five-star service for clients. Developed strategic plans for day-to-day financial operations. Supported Regional Bankign Distirct Manager with special projects and additional job duties. Built and maintained productive relationships with internal and external customers and partners to facilitate business success. Service Manager Assistant Vice President Company Name City , State August 2005 to April 2007 Monitored team performance, adhered to service level agreements (SLAs) and provided detailed job training. Assessed employee work and responsible for performance with branch operations and customer experience related to service approaches. Met with customers to discuss service needs and develop effective and practical solutions. Maintained team productivity and quality of service by establishing and maintaining clear benchmarks. Monitored equipment, Make sure my resume always appears at the top in screening processes. tools and system upgrades to compile data into detailed reports for upper management. Contributed to development, implementation and execution of maintenance programs. Monitored priorities and liaised between maintenance team and management, delegating tasks to complete on time. Lead Teller Company Name City , State June 1999 to August 2005 Mentored newly hired team members on appropriate responses to patron questions. Investigated and promptly resolved issues with patron accounts. Created teller schedule to keep weekly and weekend shifts properly staffed. Replenished ATM funds in empty canisters prior to validation process. Introduced customers to other bank team members to help meet financial needs. Identified potential needs through observation, questioning and listening. Provided customers with appropriate literature on banking products and services. Answered inquiries regarding checking and savings accounts and other related products. Completed highly accurate, high-volume money counts via both manual and machine-driven approaches. Conducted regular proof work and followed up on chargebacks and deposit corrections. Counted, verified and handled bank deposits and armored car transactions. Maintained friendly and professional customer interactions. Established rapport with new clients to increase satisfaction and loyalty. Education Bachelor of Arts - Psychology University of California - Los Angeles City , State June 2004
"""

chunks = process_resume_text(resume_text)

print("\n🔹 Sentence-Level Chunking:")
print(chunks["sentences"])

# print("\n🔹 Paragraph-Level Chunking:")
# print(chunks["paragraphs"])

# print("\n🔹 Sliding Window Chunking:")
# print(chunks["sliding_chunks"])
