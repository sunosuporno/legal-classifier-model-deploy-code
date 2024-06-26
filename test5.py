from transformers import pipeline, BartForConditionalGeneration, BartConfig
import requests
from bs4 import BeautifulSoup
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize




def document_filter(paragraphs):
    # Fetch court document from URL
    # res = requests.get(url)
    # soup = BeautifulSoup(res.text, 'html.parser')
    # text = str(soup.get_text())

    # # Tokenize sentences
    # sentences = sent_tokenize(text)


    # # Extract key facts
    # # keywords = get_keywords(text)
    # # facts = get_sentences_with_keywords(sentences, keywords)

    # # Extract decision
    pattern = r"Court (rules?|decides?|finds?|shall?|may?|must?) (.*)"
    # paragraphs = soup.find_all(["p", "blockquote", "pre"])
    
    matched_paragraphs = []
    
    for paragraph in paragraphs:
        text = paragraph.get_text()
        match = re.search(pattern, text)
        if match:
            matched_paragraphs.append(text.strip())

    pattern = r"""decide.*|conclude.*|find.*|determine.*|rule.*|hold.*|order.*|grant.*|deny.*|affirm.*|reverse.*|"the court finds that.*|it is hereby ordered that.*|the judgment is entered in favor of.*|the plaintiff's claim is dismissed.*|the defendant is held liable for.*|the appeal is denied/granted.*|based on the evidence presented, we conclude that.*|for the reasons stated above, we rule that.*|this court holds that.*|in our opinion.*|the majority opinion is.*|in accordance with precedent.*|as a result, the defendant is sentenced to.*|the contract is deemed null and void.*|the injunction is granted/preliminarily enjoined.*|the damages awarded to the plaintiff are.*|the motion for summary judgment is granted/denied.*|adjudge.*|resolve.*|pronounce.*|concur.*|dissent.*|dismiss.*|remand.*|sustain.*|overrule.*|revoke.*|invalidate.*|affirm.*|reject.*|modify.*|clarify.*|uphold.*|grant.*|relief.*|in accordance with the law.*|the court is of the opinion that.*|upon consideration of the evidence.*|having reviewed the pleadings.*|for the reasons set forth herein.*|in light of the legal principles.*|it is hereby decreed that.*|the court is persuaded by the argument.*|upon careful review of the facts.*|after due deliberation.*|in conformity with legal precedent.*|upon thorough examination of the issue.*|the court finds no merit in the claim.*|after thorough analysis of the law.*|it is the determination of this court that.*|awarded punitive damages.*|found in contempt.*|placed on probation.*|ordered to comply.*|directed to cease and desist.*|granted injunctive relief.*|dismissed for lack of standing.*|granted a writ of certiorari.*|granted a motion for reconsideration.*|directed to pay restitution.*|directed to perform community service.*|ordered to submit to a mental health evaluation.*|bail.*|laches.*|estoppel.*|res judicata.*|tortfeasor.*|proximate cause.*|contributory negligence.*|joint venture.*|bona fide purchaser.*|prima facie case.*|per curiam opinion.*|dicta.*|inter alia.*|subpoena duces tecum.*|sua sponte.*|ex parte communication.*|in camera review.*|the court is satisfied that.*|upon a thorough review of the record.*|having carefully considered the arguments.*|after weighing the evidence.*|for the reasons articulated herein.*|upon analysis of the relevant legal principles.*|in light of the case law cited.*|it is the finding of this court that.*|after thorough examination and deliberation.*|the court has taken into account the testimony of.*|based on the totality of the circumstances.*|in light of the prevailing legal standards.*|it is the judgment of this court that.*|judgments.*"""

    for paragraph in paragraphs:
        text = paragraph.get_text()
        match = re.search(pattern, text)
        if match:
            matched_paragraphs.append(text.strip())

    # Search for cybercrime sections
    section = r"66\([A-F]\)|66-[ABCDEF]|67|66|292|354-[CD]|354\([CD]\)|379|420|465|463|468"
    matched_sections = []
    for paragraph in paragraphs:
        text = paragraph.get_text()
        matches = re.findall(section, text)
        for match in matches:
            if match not in matched_sections:
                matched_sections.append(match)

            
    joined_paragraphs = " ".join(matched_paragraphs)
    summary = summarize_doc(joined_paragraphs)

    return (matched_sections, summary)
    


def summarize_doc(textString):
    config = BartConfig.from_pretrained("facebook/bart-large-cnn")

    # Update config parameters
    config.max_position_embeddings = 2048
    config.num_beams = 4
    config.length_penalty = 2.0

    # Instantiate model with updated config
    model = BartForConditionalGeneration(config=config)
    # text = ' '.join([p.get_text() for p in soup.find_all('p')])
    text = textString

    # Split text into chunks of at most 1000 tokens (padding room)
    max_len = 1000
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn") 
    chunks = [text[i:i+max_len] for i in range(0, len(text), max_len)]

    # Summarize each chunk
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk)[0]["summary_text"]
        summaries.append(summary)

    # Concatenate summary chunks  
    summary = " ".join(summaries)

    # Keep summarizing until the length of the summary is less than or equal to 1000 tokens
    while len(summary) > max_len:
        new_chunks = [summary[i:i+max_len] for i in range(0, len(summary), max_len)]
        new_summaries = []
        for chunk in new_chunks:
            new_summary = summarizer(chunk)[0]["summary_text"]
            new_summaries.append(new_summary)
        summary = " ".join(new_summaries)

    # Final summarization
    final_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    final_summary = final_summarizer(summary, min_length=150, max_length=250)[0]["summary_text"]
    print('Final Summary:' + final_summary)
    return final_summary



