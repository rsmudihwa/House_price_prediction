import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

# load dataset
prop = pd.read_csv("Properties.csv")

# Renaming some cities properly to aid our prediciton perform better
house['city'].replace('North', 'Isheri North', inplace=True)
house['city'].replace('Island', 'Lagos Island', inplace=True)

# Creating island features of places from location columns 
def island_flag(house):
    if (house['city']=='Lekki') or (house['city']=='Victoria Island') or (house['city']=='Ajah') \
     or (house['city']=='Ikoyi') or (house['city']=='Eko Atlantic City') \
    or (house['city']=='Epe') or (house['city']=='Lagos Island'):
        return 1
    else:
        return 0

# Create Island features
house['island_flag'] = house.apply(island_flag, axis=1)

# Create Estate feature
house['estate_flag'] = house['location'].apply(lambda x: len([c for c in str(x).lower().split() if "estate" in c]))

# Create terrace/duplex/detached houses feature
house['terrace_flag'] = house['title'].apply(lambda x: len([c for c in str(x).lower().split() if "terraced" 
                                                          in c or "terrace" in c or "detached" in c
                                                                 or "duplex" in c]))

# Create new/renovated houses feature
house['new_flag'] = house['title'].apply(lambda x: len([c for c in str(x).lower().split()
                                                              if "new" in c or "newly" in c or "renovated" in c]))

# Create luxurious/executive/exquisite houses feature
house['luxury_flag'] = house['title'].apply(lambda x: len([c for c in str(x).lower().split()
                                                              if "luxury" in c or "executive" in c or "luxurious" in c
                                                        or "exquisite"in c or "excellent" in c]))

# Create Serviced houses feature
house['serviced_flag'] = house['title'].apply(lambda x: len([c for c in str(x).lower().split()
                                                                           if "serviced" in c]))

# Create studio apartment feature
house['studio_flag'] = house['title'].apply(lambda x: len([c for c in str(x).lower().split()
                                                                           if "studio" in c]))

# Create big/spacious houses feature
house['big_flag'] = house['title'].apply(lambda x: len([c for c in str(x).lower().split() if "spacious" in c
                                                     or "big" in c]))

# Create self-contain houses feature
house['selfcontain_flag'] = house['description'].apply(lambda x: len([c for c in str(x).lower().split() if "self" in c]))


# Create flat feature
house['flat_flag'] = house['description'].apply(lambda x: len([c for c in str(x).lower().split() if "flat" in c]))

# Droping all office properties
house = house[~(house['title'].str.contains('office', case=False))]

# Dropping irrelevant features
house2 = house.house(['page','title', 'description', 'location','garage'], axis='columns')

house4 = house2[~(house2['price']>house2['price'].quantile(0.95))]

house5 = house4[~(house4['bathrooms'] >house4['bathrooms'].quantile(0.999))]


house6 = house5[~(house5['bathrooms']>house5['bedrooms']+2)]

house7 = house6[~(house6['bathrooms']>house6['toilets'])]

house8 = house7[~(house7['toilets'] >house7['bedrooms']+2)]

dummies = pd.get_dummies(house8['city'], drop_first=True)

house9 = pd.concat([house8, dummies], axis=1).drop('city',axis=1)

seed = 42
np.random.seed(seed)

X = house9.drop(['price'], axis=1)
y = house9['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

catboost = CatBoostRegressor(depth=10, iterations=1000, learning_rate=0.01, random_state=seed, verbose=False)
catboost.fit(X_train, y_train)

with open('Lagos_properties_price_model.pickle','wb') as f:
    pickle.dump(catboost, f)

pickle_in = open('Lagos_properties_price_model.pickle', 'rb')
model = pickle.load(pickle_in)

def predict_price(city,bedroom,bathroom,toilet,terraced=False,self_contain=False,flat=False):
    loc_index = np.where(X.columns==city)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = bedroom
    x[1] = bathroom
    x[2] = toilet
    x[5] = terraced
    x[11] = self_contain
    x[12] = flat
    if loc_index >= 0 :
        x[loc_index] = 1
        
    return '₦{:,}'.format(int(model.predict([x])[0]))

def  main():
    st.title('House Price Prediction Project')

    html_temp = """
    <div style="background-color:#f63366;
        border-radius: 25px;
        padding:5px">
    <h2 style="color:white;
        text-align:center;">Lagos House Prices Prediciton ML APP</h2>
    </div>
    """
    
    st.markdown(html_temp, unsafe_allow_html=True)
    city = st.selectbox('Choose a Location',('Lekki','Ikoyi', 'Victoria Island', 'Ajah', 'Isheri North',
       'Maryland', 'Ogudu', 'Ikeja', 'Ikorodu', 'Shomolu','Yaba',
       'Gbagada', 'Magodo', 'Ilupeju', 'Surulere', 'Ketu', 'Odofin',
       'Ojodu', 'Ojota', 'Kosofe', 'Agege', 'Isolo', 'Mushin', 'Ijede',
       'Ayobo', 'Ipaja', 'Ikotun', 'Eko Atlantic City', 'Alimosho', 'Ojo',
       'Ifako-Ijaiye', 'Ijaiye', 'Epe', 'Lagos Island', 'Oshodi', 'Idimu',
       'Apapa'))
    bedrooms = st.selectbox('Number of Bedrooms',(0,1,2,3,4,5))
    bathrooms = st.selectbox('Number of Bathrooms',(1,2,3,4,5))
    toilets = st.selectbox('Number of Toilets',(1,2,3,4,5))
    
    st.write("Do you want a Self-contain/Flat/Terraced Apartment?")
    terraced = st.radio('Terraced?',(True,False))
    flat = st.radio('Flat?',(True,False))
    self_contain = st.radio('Self Contain?',(True,False))
 
    if (terraced == False):
        st.text('Not a Terraced')
    if (flat == False):
        st.text('Not a Flat')
    if (self_contain == False):
        st.text('Not a Self-contain')
    
        
    st.write('You selected:','\n' 
             'Location:',city,'\n' 
             '\n Bedroom:',bedrooms,'\n' 
             '\n Bathroom:',bathrooms,'\n' 
             '\n Toilet:',toilets,'\n'
    )
    result = ""
    
    if st.button('Predict Price'):
        result = predict_price(city,bedrooms,bathrooms,toilets,terraced,flat,self_contain)
    st.success('The price is {}'. format(result))
    if st.button('About'):
        st.text('Adebo Dolapo')
        
    
if __name__=='__main__':
    main()