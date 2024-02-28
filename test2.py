# import requests
# from transformers import AutoTokenizer, BartForConditionalGeneration 

# model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# url = "https://indiankanoon.org/doc/298882/"
# page = requests.get(url).text

# ARTICLE_TO_SUMMARIZE = page
# max_length = 16384

# inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=max_length, return_tensors="pt", truncation=True)

# # Generate Summary  
# summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=512)
# print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])


# from transformers import AutoTokenizer, BartForConditionalGeneration

# model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# ARTICLE_TO_SUMMARIZE = (
#     "These opposite parties further contended that the complainant informed the insurer i.e. Bank Assurance Division, through opposite party no.1, about the fire accident on 3.9.2010. The Bank Assurance Division deputed their surveyor and loss assessor to assess the net loss. The said surveyor requested the complainant to furnish all necessary documents including the policy copy. Inspite of several requests, the complainant did not furnish the relevant documents except the policy copy. The surveyor, on observing the policy, found that the policy was Burglary BP Policy covering only oil and lubricants, which is not relevant to the peril. The risk is supposed to be covered under Standard Fire and Special Perils Policy. The surveyor submitted his report on 7.2.2011 stating that the occurrence of loss or damage to the stocks of the insured is purely accidental in nature, i.e. due to short circuit, which is cause of peril, not covered under/or fall within the scope of Burglary BP policy, as such there is no liability on the part of the insurer. Therefore, the file was closed as no claim and informed the same to the complainant through letter dt.4.3.2011. They further contended that they have not received any instructions from the opp.party no.1 or the complainant to insure the stocks against the fire risks, as such burglary policy was renewed and there was no fire policy."
# )
# inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

# # Generate Summary
# summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=40)
# print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

# from transformers import AutoTokenizer, BartForConditionalGeneration

# model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# # Read content from file
# with open("doc.txt", "r", encoding="utf-8") as file:
#     ARTICLE_TO_SUMMARIZE = file.read()

# # Increase max_length to 2048
# inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=2048, return_tensors="pt", truncation=True)

# # Generate Summary
# summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=100)
# summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# print(summary)


from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-pubmed")

# by default encoder-attention is `block_sparse` with num_random_blocks=3, block_size=64
model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-pubmed")

# decoder attention type can't be changed & will be "original_full"
# you can change `attention_type` (encoder only) to full attention like this:
model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-pubmed", attention_type="original_full")

# you can change `block_size` & `num_random_blocks` like this:
model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-pubmed", block_size=16, num_random_blocks=2)

text = "On the other hand, learned counsel for the State has opposed the prayer for bail. It is submitted that the appellant- Mahesh Kumar Poddar along with four others namely Sonu Kumar Mahto, Pradip Mazumdar, Mantosh Kumar Poddar and Ritesh Kumar have been rightly convicted by the learned Court blow. It is submitted that co-convict, Sonu Kumar Mahato was having found dubious bank account, which was verified and proved by Sanjay Kumar Sinha, Branch Manager, Indian Bank, Jamshedpur and thereafter Pradip Mazumdar, Mantosh Kumar Poddar were arrested and the name of the appellant Mahesh Kumar Poddar has come in this case on the basis of confessional statement of co-convict Pradip Mazumdar. It is submitted that co- convict Pradip Mazumdar is the mastermind of the entire crime and the appellant- Mahesh Kumar Poddar happens to be the associate of the said Mantosh Kumar Poddar and Pradip Mazumdar and Sonu Kumar Mahato and it is evident from the confessional statement of co-convicts Pradip Mazumdar and Sonu Kumar Mahato marked as Ext.-14 and Ext. 11 respectively. It is submitted that series of documents i.e. 16 ATM Cards, nine passbooks of different banks, five mobile phones, 43 PAN cards, 16 SIM cards, cheque books, various I.D. proofs etc. have been recovered from the possession of co-convict Sonu Kumar Mahato and Pradip Mazumdar respectively and all these above documents fully established the involvement of the present appellant - Mahesh Kumar Poddar along with four others namely Sonu Kumar Mahato, Pradip Mazumdar, Mantosh Kumar Poddar and Ritesh Kumar in cheating the general people. It is submitted that the appellant during his statement recorded under Section 313 of the Cr. P. C., has also admitted that the bank account belongs to him, however the appellant could not give proper reply from which source sum of Rs. 21,85,037/- had come from the different accounts holders and which was credited in the account of the appellant on various dates through online mode from various districts all over the country. It is submitted that P.W.-2, Sanjay Kumar Sinha and P.W.-9, Upendra Kumar Mahato, I.O., had fully supported the prosecution case. It is submitted that P.W.-9 is the I.O. of this case and who has brought on record Ext. 26 and 26(1) respectively, which are the bank account statements of the appellant and duly certified under Section 65-B of the Indian Evidence Act, which go to prove the guilt."
inputs = tokenizer(text, return_tensors='pt')
prediction = model.generate(**inputs)
prediction = tokenizer.batch_decode(prediction)