# MLproject
4N (Negative News Neural Nets) - The Adverse Media Project

The Problem
All financial organizations need to do compliance investigations on their customers. Looking for adverse media ( https://complyadvantage.com/knowledgebase/adverse-media/ ), also known as negative news screening on people and organizations who are our customers aids in these investigations.

Looking for adverse media is expensive - be it paying for specialists to search and read through the news on SERP (Search Engine Results Page) manually or paying for an external API to do the same automatically. 

Even more importantly, as any whitepaper or marketing article of any adverse media api will tell you, since manual adverse media checks are slow, they might miss the important news article on page 42 of the search engine results. This would result in a subpar fight against financial crime by allowing unwanted entities to move money across the globe. 

The Goal
The only way for fintech startups and others to grow scalably is to build a solution to do these checks automatically in-house. This project is the start of a long-running effort towards that solution. The final solution of this project will provide a shortlist of negative news on a given organization / person name.

Proposed Methods
Detect if a given text is adverse media and if the given text is about a given name using NLP. Also, provide the words from the text which made the model come to the provided conclusion.

To be able to train adverse media detection, you need training data. You will be provided software engineers who will help to gather it and a small budget to pay for API-s.


Plan

Find and validate adverse media sources to scrape. Here's some examples:
https://offshoreleaks.icij.org/
https://www.riskscreen.com/kyc360/news/malta-five-remittance-agents-investigated-for-money-laundering/
https://www.securitieslawyer101.com/2020/sec-charges-attorney-ben-bunker-with-fraudulent-scheme/
Decide if this approach of scraping for adverse media news from specific news sites to gather training data is the right way to go.  (I have started another in-house training data gathering initiative which should result in some hundreds/thousands of urls to adverse media news urls, which would be great data to train on.)
Scrape the news or get these some other way - Maybe use some SERP API to google for adverse news using some specific keywords and then scrape the results
Train a model to distinguish if a news story is about adverse media or not
Maybe also train a model to return the names the adverse media is about. Or consider some other solution where the name of the person we are looking adverse media for is known.





