# Overview of Bryan's Result
- I have successfully scraped a total of 1,835 blogs and articles. You can check this in the folder `scraped_articles`
- I perform a random sampling of 322 websites out of 1,835 and classify their relevance to our topic. From the random sampling, I found that 258 (80.1%) of them are relevant. Among the relevant ones, I found that 242/258 (93.8%) of them conveys an opinion/sentiment towards the topic. The result of this random sampling and classification can be seen in `relevant_checkpoint.json` 

# Methodology
1. Crawling 
    - I perform crawling in three stages:
        1. In the first stage, please refer to the folder `articles_extraction_old`
            - In this stage, I use the Serper search API to a search a list of keywords from certain websites that I listed. I take the search result and add the URLs into a list. 
        2. In the second stage, please refer to the folder `articles_extraction`
            - In this stage, I use Gemini CLI to automatically crawl the website for relevant blogs and articles. I provided the CLI with tools (see `server.py`) and detailed instruction on what to find (see `GEMINI.md`). It took around 1 hour to crawl all the websites. 
        3. The third stage is just combining all of the results into one. See `merged_checkpoints.md` 
2. Scraping
    - After I obtained the list of URLs, I scrape the content. Please see `scraped_articles.py` for more details. 
    - Unfortunately, not all URLs can be scraped. From over 2000 crawled URLs, only 1,835 contains valid article content. 
3. Classification
    - 1,835 is a huge number to classify. Since we are talking about articles, each could potentially contain thousands of words. This means simple Transformer (e.g., Small Language Model) wouldn't be able to reliably classify the articles. 
    - So, instead of trying to classify all of them and remove irrelevant materials right away, I perform random sampling of 322 scraped contents and classify them. I found out that 80.1% of them are relevant. I am unsure if this threshold is acceptable, this is something to be discussed. 



    
