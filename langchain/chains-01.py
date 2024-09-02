# DESCRIPTION
#   Lanchain - Chains
#   Play with various prompts to see the outputs
#
#   GGUF and GGML are file formats used for storing models for inference,
#   especially in the context of language models like GPT (Generative Pre-trained Transformer).
#
# install packages langchain, langchain-community, llama-cpp-python
#
# Last Updated: 2024-09-02

from langchain_community.llms import LlamaCpp

from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate

# enable verbose to debug the LLM's operation
verbose = False

model_path = "/Users/jbarozet/LLM/llama-2-7b-chat.Q4_K_M.gguf"
# model_path="/Users/jbarozet/LLM/synthia-7b-v2.0-16k.Q4_K_M.gguf"

# With CPU
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.75,
    max_tokens=2048,
    n_ctx=2048,
    top_p=1,
    verbose=verbose,
)

article = """Coinbase, the second-largest crypto exchange by trading volume, released its Q4 2022 earnings on Tuesday, giving shareholders and market players alike an updated look into its financials. In response to the report, the company's shares are down modestly in early after-hours trading.In the fourth quarter of 2022, Coinbase generated 2.49 billion in the year-ago quarter. Coinbase's top line was not enough to cover its expenses: The company lost 2.46 per share, and an adjusted EBITDA deficit of 581.2 million in revenue and earnings per share of - 201.8 million driven by 8.4 million monthly transaction users (MTUs), according to data provided by Yahoo Finance.Before its Q4 earnings were released, Coinbase's stock had risen 86% year-to-date. Even with that rally, the value of Coinbase when measured on a per-share basis is still down significantly from its 52-week high of 26 billion in the third quarter of last year to 133 billion to 1.5 trillion during 2022, which resulted in Coinbase's total trading volumes and transaction revenues to fall 50% and 66% year-over-year, respectively, the company reported.As you would expect with declines in trading volume, trading revenue at Coinbase fell in Q4 compared to the third quarter of last year, dipping from 322.1 million. (TechCrunch is comparing Coinbase's Q4 2022 results to Q3 2022 instead of Q4 2021, as the latter comparison would be less useful given how much the crypto market has changed in the last year; we're all aware that overall crypto activity has fallen from the final months of 2021.)There were bits of good news in the Coinbase report. While Coinbase's trading revenues were less than exuberant, the company's other revenues posted gains. What Coinbase calls its "subscription and services revenue" rose from 282.8 million in Q4 of the same year, a gain of just over 34% in a single quarter.And even as the crypto industry faced a number of catastrophic events, including the Terra/LUNA and FTX collapses to name a few, there was still growth in other areas. The monthly active developers in crypto have more than doubled since 2020 to over 20,000, while major brands like Starbucks, Nike and Adidas have dived into the space alongside social media platforms like Instagram and Reddit.With big players getting into crypto, industry players are hoping this move results in greater adoption both for product use cases and trading volumes. Although there was a lot of movement from traditional retail markets and Web 2.0 businesses, trading volume for both consumer and institutional users fell quarter-over-quarter for Coinbase.Looking forward, it'll be interesting to see if these pieces pick back up and trading interest reemerges in 2023, or if platforms like Coinbase will have to keep looking elsewhere for revenue (like its subscription service) if users continue to shy away from the market.
"""

# --[[ Basic LLMChain - Fact Extraction ]]

fact_extraction_prompt = PromptTemplate(
    input_variables=["text_input"],
    template="Extract the key facts out of this text. Don't include opinions. Give each fact a number and keep them short sentences. :\n\n {text_input}",
)

fact_extraction_chain = LLMChain(llm=llm, prompt=fact_extraction_prompt, verbose=False)
facts = fact_extraction_chain.invoke(article)

print(facts["text"])

# --[[ Rewrite as a summary from the facts]]

investor_update_prompt = PromptTemplate(
    input_variables=["facts"],
    template="You are a Goldman Sachs analyst. Take the following list of facts and use them to write a short paragrah for investors. Don't leave out key info:\n\n {facts}",
)

investor_update_chain = LLMChain(llm=llm, prompt=investor_update_prompt)
investor_update = investor_update_chain.invoke(facts["text"])

print(investor_update)
len(investor_update)

# --[[ Chaining these together ]]

full_chain = SimpleSequentialChain(
    chains=[fact_extraction_chain, investor_update_chain], verbose=True
)
response = full_chain.invoke(article)

print(response)
