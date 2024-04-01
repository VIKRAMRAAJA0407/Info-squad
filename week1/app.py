import numpy as np
import pandas as pd
import streamlit as st

def learn(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]
    
    for i, h in enumerate(concepts):
        if target[i] == "Yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'

        if target[i] == "No":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])
    
    return specific_h, general_h

def main():
    st.title("Candidate Elimination Algorithm")

    # Section to get input from the user
    st.header("Input Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Input Data:")
        st.write(data)

        # Separating concept features from Target
        concepts = np.array(data.iloc[:,0:-1])
        target = np.array(data.iloc[:,-1])

        # Call the learn function
        s_final, g_final = learn(concepts, target)

        # Display final specific and general hypotheses in tabular form
        st.header("Final Specific Hypothesis")
        st.write(pd.DataFrame([s_final]))

        st.header("Final General Hypothesis")
        st.write(pd.DataFrame(g_final))

if __name__ == "__main__":
    main()
