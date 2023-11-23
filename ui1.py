import streamlit as st
import pandas as pd
from AI_Chat_Bot import nvidia_api_call, api_key, invoke_url, fetch_url_format
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma 

def process_user_query(user_query):
    df_ret= pd.read_csv('/Users/mrunmayeerane/Desktop/progress/Flavors/gitupload/Retrival based/aggregated_data.csv')

  
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"
    model_kwargs = {'device':'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(model_name=modelPath, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    db3 = Chroma(embedding_function=embeddings, persist_directory="/Users/mrunmayeerane/Desktop/progress/Flavors/chroma_db_1")
    docs = db3.similarity_search(user_query,5)[::-1]

    combined_reviews = ""

    for i in range(len(docs)):
        row_value = docs[i].metadata.get('row', None)
        Name = df_ret.iloc[row_value]['name']
        Address = df_ret.iloc[row_value]['address']
        City = df_ret.iloc[row_value]['city']
        State = df_ret.iloc[row_value]['state']
        PostalCode = df_ret.iloc[row_value]['postal_code']
        Latitude = df_ret.iloc[row_value]['latitude']
        Longitude = df_ret.iloc[row_value]['longitude']
        Stars = df_ret.iloc[row_value]['stars_y']
        ReviewsCount = df_ret.iloc[row_value]['review_count']
        IsOpen = df_ret.iloc[row_value]['is_open']
        Hours = df_ret.iloc[row_value]['hours']
        Text = df_ret.iloc[row_value]['user_reviews']

        # Create the review prompt for each business
        review_prompt = f"Business Review:\n"
        review_prompt += f"Name: {Name}\n" if Name else ""
        review_prompt += f"Address: {Address}, {City}, {State}, {PostalCode}\n" if Address and City and State and PostalCode else ""
        review_prompt += f"Hours: {Hours}\n" if Hours else ""
        review_prompt += f"Rating: {Stars} stars\n" if Stars else ""
        # review_prompt += f"Text: {Text}\n" if Text else ""
        
    
    
        combined_reviews += review_prompt + "\n"
    final_prompt = combined_reviews + "You are a smart recommender system, Please provide a recommendation based on this  information.\nRecommend places from suggested additional context only and from file aggregated_data.csv \nDo not suggest places on your own\n Do not mention aggregated_data.csv file in your response and your response must suggest all Business Reviews included in prompt"
    print(final_prompt)
    response = nvidia_api_call(final_prompt, api_key, invoke_url, fetch_url_format)
    return response

def main():
    st.title("Food Recommendation Chatbot")

   
    if "messages" not in st.session_state:
        st.session_state.messages = []

  
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

 
    if user_query := st.chat_input("Enter your query:"):
       
        st.session_state.messages.append({"role": "user", "content": user_query})

     
        response = process_user_query(user_query)

       
        st.session_state.messages.append({"role": "assistant", "content": response})

      
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

if __name__ == "__main__":
    main()

