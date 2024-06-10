import re
import requests
import nltk
nltk.download('punkt')
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize  
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize


def summarize_court_doc(url, word_count=350):

    # Fetch court document from URL
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')
    text = soup.get_text()

    # Tokenize sentences
    sentences = sent_tokenize(text)

    # Extract issues using regex
    issues = re.findall(r"The ([a-z]+)\s+issues? presented are: (.*)", text)
    if issues:
        issues = issues[0][1].split(", ")
    else:
        issues = []

    # Extract key facts 
    # keywords = get_keywords(text)
    # facts = get_sentences_with_keywords(sentences, keywords)

    # Extract decision
    pattern = r"Court (rules?|decides?|finds?|shall?|may?|must?) (.*)"
    paragraphs = soup.find_all("p")
    for paragraph in paragraphs:
        text = paragraph.get_text()
        match = re.search(pattern, text)
        if match:
            print(f"Matched text: {match.group()}")
            print(f"Paragraph text: {text.strip()}")
            print()

    judgement = re.search(r"""decide.*|conclude.*|find.*|determine.*|rule.*|hold.*|order.*|grant.*|deny.*|affirm.*|reverse.*|"the court finds that.*|it is hereby ordered that.*|the judgment is entered in favor of.*|the plaintiff's claim is dismissed.*|the defendant is held liable for.*|the appeal is denied/granted.*|based on the evidence presented, we conclude that.*|for the reasons stated above, we rule that.*|this court holds that.*|in our opinion.*|the majority opinion is.*|in accordance with precedent.*|pursuant to [statute/case law].*|as a result, the defendant is sentenced to.*|the contract is deemed null and void.*|the injunction is granted/preliminarily enjoined.*|the damages awarded to the plaintiff are.*|the motion for summary judgment is granted/denied.*|adjudge.*|resolve.*|pronounce.*|concur.*|dissent.*|dismiss.*|remand.*|sustain.*|overrule.*|revoke.*|invalidate.*|affirm.*|reject.*|modify.*|clarify.*|uphold.*|grant.*|relief.*|in accordance with the law.*|the court is of the opinion that.*|upon consideration of the evidence.*|having reviewed the pleadings.*|for the reasons set forth herein.*|in light of the legal principles.*|it is hereby decreed that.*|the court is persuaded by the argument.*|upon careful review of the facts.*|after due deliberation.*|in conformity with legal precedent.*|upon thorough examination of the issue.*|the court finds no merit in the claim.*|after thorough analysis of the law.*|it is the determination of this court that.*|awarded punitive damages.*|found in contempt.*|placed on probation.*|ordered to comply.*|directed to cease and desist.*|granted injunctive relief.*|dismissed for lack of standing.*|granted a writ of certiorari.*|granted a motion for reconsideration.*|directed to pay restitution.*|directed to perform community service.*|ordered to submit to a mental health evaluation.*|bail.*|laches.*|estoppel.*|res judicata.*|tortfeasor.*|proximate cause.*|contributory negligence.*|joint venture.*|bona fide purchaser.*|prima facie case.*|per curiam opinion.*|dicta.*|inter alia.*|subpoena duces tecum.*|sua sponte.*|ex parte communication.*|in camera review.*|the court is satisfied that.*|upon a thorough review of the record.*|having carefully considered the arguments.*|after weighing the evidence.*|for the reasons articulated herein.*|upon analysis of the relevant legal principles.*|in light of the case law cited.*|it is the finding of this court that.*|after thorough examination and deliberation.*|the court has taken into account the testimony of.*|based on the totality of the circumstances.*|in light of the prevailing legal standards.*|it is the judgment of this court that.*|judgments.*""", text)


#     # Construct summary  
#     summary = ""
#     summary += construct_section("Issues", issues)
#     summary += construct_section("Facts", facts)
#     summary += construct_section("Decision", [decision])

#     # Shorten summary 
#     words = summary.split()
#     summary = " ".join(words[:word_count])
    
#     return summary

# def get_keywords(text):
#     no_stopwords = [word for word in word_tokenize(text) if not word in stopwords.words('english')]
#     stemmed_words = [PorterStemmer().stem(word) for word in no_stopwords]
#     return sorted(set(stemmed_words), key=stemmed_words.count, reverse=True)[:10]

# def get_sentences_with_keywords(sentences, keywords):
#     keyword_sentences = []
    
#     facts_intro = re.compile(r"The brief facts of the case is as follows|The brief facts in|The facts are")
    
#     in_facts = False
#     for sentence in sentences:
#         if re.search(facts_intro, sentence):
#             in_facts = True
            
#         if in_facts:
#             for word in keywords:
#                 if re.search(r"\b"+word+r"\b", sentence):
#                     keyword_sentences.append(sentence)
#                     break
   
#     return keyword_sentences

# def construct_section(title, content):
#     section = f"# {title}\n"
#     section += ". ".join(content) + "\n\n" 
#     return section

# Sample usage        
url = "https://indiankanoon.org/doc/107331326/"
# print(summarize_court_doc(url))
summarize_court_doc(url)




# # Define the regex pattern for identify judgement present in case docs
# judgement_pattern=r"""decide.*|conclude.*|find.*|determine.*|rule.*|hold.*|order.*|grant.*|deny.*|affirm.*|reverse.*|"the court finds that.*|it is hereby ordered that.*|the judgment is entered in favor of.*|the plaintiff's claim is dismissed.*|the defendant is held liable for.*|the appeal is denied/granted.*|based on the evidence presented, we conclude that.*|for the reasons stated above, we rule that.*|this court holds that.*|in our opinion.*|the majority opinion is.*|in accordance with precedent.*|pursuant to [statute/case law].*|as a result, the defendant is sentenced to.*|the contract is deemed null and void.*|the injunction is granted/preliminarily enjoined.*|the damages awarded to the plaintiff are.*|the motion for summary judgment is granted/denied.*|adjudge.*|resolve.*|pronounce.*|concur.*|dissent.*|dismiss.*|remand.*|sustain.*|overrule.*|revoke.*|invalidate.*|affirm.*|reject.*|modify.*|clarify.*|uphold.*|grant.*|relief.*|in accordance with the law.*|the court is of the opinion that.*|upon consideration of the evidence.*|having reviewed the pleadings.*|for the reasons set forth herein.*|in light of the legal principles.*|it is hereby decreed that.*|the court is persuaded by the argument.*|upon careful review of the facts.*|after due deliberation.*|in conformity with legal precedent.*|upon thorough examination of the issue.*|the court finds no merit in the claim.*|after thorough analysis of the law.*|it is the determination of this court that.*|awarded punitive damages.*|found in contempt.*|placed on probation.*|ordered to comply.*|directed to cease and desist.*|granted injunctive relief.*|dismissed for lack of standing.*|granted a writ of certiorari.*|granted a motion for reconsideration.*|directed to pay restitution.*|directed to perform community service.*|ordered to submit to a mental health evaluation.*|bail.*|laches.*|estoppel.*|res judicata.*|tortfeasor.*|proximate cause.*|contributory negligence.*|joint venture.*|bona fide purchaser.*|prima facie case.*|per curiam opinion.*|dicta.*|inter alia.*|subpoena duces tecum.*|sua sponte.*|ex parte communication.*|in camera review.*|the court is satisfied that.*|upon a thorough review of the record.*|having carefully considered the arguments.*|after weighing the evidence.*|for the reasons articulated herein.*|upon analysis of the relevant legal principles.*|in light of the case law cited.*|it is the finding of this court that.*|after thorough examination and deliberation.*|the court has taken into account the testimony of.*|based on the totality of the circumstances.*|in light of the prevailing legal standards.*|it is the judgment of this court that.*|judgments.*"""