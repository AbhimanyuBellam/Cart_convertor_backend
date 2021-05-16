# from flask import Flask,jsonify,request
# from flask_cors import CORS, cross_origin
# import pandas as pd
# df = pd.DataFrame(columns=['data'])
# arr=[]
# app = Flask(__name__)
# CORS(app)
# @app.route('/')
# def hello():
#     return jsonify({'text':'Hello World!'})


# @app.route('/mouse', methods=['POST', 'GET'])
# def recv_mouse():
#      # if(request.method=="POST"):
#      mouse_data = request.get_json(force=True)
#      # if(data=={}):
#      # df['data']=arr
#      # df.to_csv("digital_behavior.csv", index=False)
#      # return data
#      # arr.append(data)
#      print(mouse_data)
#      return(mouse_data)
#      # else:
#      # return jsonify(data)

# @app.route('/cart',methods=['POST','GET'])
# def recv_data():
#      # if(request.method=="POST"):
#      data=request.get_json(force=True)
#      # if(data=={}):
#           # df['data']=arr
#           # df.to_csv("digital_behavior.csv", index=False)
#           # return data
#      # arr.append(data)
#      print(data)
#      return(data)
#      # else:
#           # return jsonify(data)


# if __name__ == '__main__':
#      app.run(host='0.0.0.0')
# from flask import Flask, jsonify, request
# from flask_cors import CORS, cross_origin
# import pandas as pd
# df = pd.DataFrame(columns=['data'])
# arr = []
# app = Flask(__name__)
# CORS(app)


# @app.route('/')
# def hello():
#     return jsonify({'text': 'Hello World!'})


# @app.route('/hello', methods=['POST'])
# def recv_data():
#      data = request.get_json()
#      # if(data == {}):
#      #      df['data'] = arr
#      #      df.to_csv("digital_behavior.csv", index=False)
#      #      return data
#      # arr.append(data)
#      print(data)
#      # print(arr)
#      return jsonify(data)


# if __name__ == '__main__':
#      app.run(port=5002)
# from flask import Flask,jsonify,request
# from flask_cors import CORS, cross_origin
# import pandas as pd
# df = pd.DataFrame(columns=['data'])
# arr=[]
# app = Flask(__name__)
# CORS(app)
# @app.route('/')
# def hello():
#     return jsonify({'text':'Hello World!'})


# # @app.route('/hello',methods=['POST','GET'])
# # def recv_data():
# #      # if(request.method=="POST"):
# #      data=request.get_json(force=True)
# #           # if(data=={}):
# #                # df['data']=arr
# #                # df.to_csv("digital_behavior.csv", index=False)
# #                # return data
# #           # arr.append(data)
# #           # print(data)
# #           # print(arr)
# #      return data
# #      # else:
# #           # return jsonify(data)
# @app.route('/hello',methods=['POST','GET'])
# def recv_data():
#      # if(request.method=="POST"):
#      data=request.get_json(force=True)
#      if(data=={}):
#           df['data']=arr
#           df.to_csv("digital_behavior_test_neg.csv", index=False)
#           return data
#      arr.append(data)
#      print(data)
#      # print(arr)
#      return data
#      # else:
#           # return jsonify(data)

# if __name__ == '__main__':
#      app.run(host='0.0.0.0')



from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np 
# import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#--------------------------------------------------------
import tensorflow as tf
global graph,model
graph = tf.get_default_graph()
from keras.models import load_model
# import pandas as pd
import csv
# from numpy import mean
# from numpy import std
# from numpy import dstack
# from keras.models import Sequential
# from keras.layers import Dense,Flatten
# from keras.layers import Dropout
# from keras.layers import LSTM
# from keras.utils import to_categorical
# from numpy import array
import math
# import os
import random
# import numpy as np

# df=pd.read_csv("neg_inst_mouse_test.csv")

#print(df["data"])
#print ("____________")
import ast
#print(ast.literal_eval(df["data"][0]))

def take_ending_entries(arr,how_many):
    return arr[len(arr)-how_many:]

def split_it(diction):
     x1=diction["X"][:40]
     x2=diction["X"][40:]
     y1=diction["Y"][:40]
     y2=diction["Y"][40:]

     return list(zip(x1,y1)),list(zip(x2,y2))

def get_x_y(pos_dict):
    positions_list=[]
    # count=0
    for i in range(len(pos_dict["X"])):
        #print (len(df["data"][i]))
     #    pos_dict=(ast.literal_eval(df["data"][i]))
        #print (len(pos_dict['X']))
     #    if len(pos_dict['X'])<40:
     #        print ("bitch")
     #        continue
        x=take_ending_entries(pos_dict['X'],40)
        #x.append(yes_no)
        #print ("x",x)
        y=take_ending_entries(pos_dict['Y'],40)
        #y.append(yes_no)
        positions_list.append(list(zip(x,y)))
        #positions_list.append([1,1])
    #print ("count",count)
    return positions_list


def make_slopes(trainX):
    trainX_new=[]
    for i in range(len(trainX)):
        j=0
        temp=[]
        slope=0
        while j+1<(len(trainX[i])):
            try:
                slope=(trainX[i][j][1]-trainX[i][j+1][1])/(trainX[i][j+1][0]-trainX[i][j][0])
            except:
                if slope==np.inf:
                    slope=10
                if slope==-np.inf:
                    slope=-10
            if math.isnan(slope):
                slope=0
            j=j+1
            

            temp.append(slope)
        trainX_new.append(temp)
    return trainX_new

def make_3d(arr):
    new_list=[]
    i=0
    while i+2<=len(arr):
        #indexes=[i+j for j in range(pair_how_many)]
        new_list.append([arr[i]]+[arr[i+1]])
        i=i+2
    return new_list

model = load_model('lstm_model.h5') #----------------
#-----------------------------------------------------------

dict_names={'Westside': 'c1', 'Zara': 'c2', 'Marks and spencers': 'c3', 'US Polo': 'c4', 'Armani': 'c5', 'Benetton': 'c6', 'Splash': 'c7', 'FCUK': 'c8', 'Wipro ': 'f1', 'Ikea': 'f2', 'Hulsta': 'f3', 'Style Spa': 'k8', 'Nilkamal': 'f5', 'Damro': 'f6', 'Durian': 'f7', 'Usha': 'f8', 'Montblanc': 'p1', 'Parker': 'p2', 'Linc': 'p3', 'Flair': 'p4', 'Reynolds': 'p5', 'Cello': 'p6', 'Camlin': 'p7', 'Classmate': 'p8', 'Puma': 's1', 'Nike': 's2', 'Reebok': 's3', 'Asics': 's4', 'Adidas': 's5', 'Head': 's6', 'Wilson': 's7', 'Skechers': 's8', 'Loreal': 'b1', 'Unilever': 'b2', 'Estee Lauder': 'b3', 'Coty': 'b4', 'Shiseido': 'b5', 'Beiersdorf': 'b6', 'Jhonson and Jhonson': 'b7', 'Amore Pacific': 'b8', 'Emporio Armani': 'w1', 'Fozzil': 'w2', 'Timex': 'w3', 'Titan': 'w4', 'Rado': 'w5', 'Tag Huer': 'w6', 'Casio': 'w7', 'Fastrack': 'w8', 'One Plus': 'm1', 'Apple': 'm2', 'Samsung': 'm3', 'Poco': 'm4', 'Redmi': 'm5', 'Huawei': 'm6', 'Oppo': 'm7', 'Vivo': 'm8', 'System76': 'l1', 'Toshiba': 'l2', 'Razor Blade': 'l3', 'Fujitsu': 'l4', 'HP': 'l5', 'Lenovo': 'l6', 'Alienware': 'l7', 'Asus': 'l8', 'Sleek': 'k1', 'Jhonson Kitchens': 'k2', 'Hafele': 'k3', 'Haecker': 'k4', 'Kohler': 'k5', 'ECBO': 'k6', 'Godrej Interio': 'k7', 'Auto international': 'a1', 'Ayushi Engineering Company': 'a2', 'Arjan Industries': 'a3', 'Best Forgings India': 'a4', 'Bomrah Industries': 'a5', 'Chunho Tech': 'a6', 'Canara Standard': 'a7', 'Cyner Industrial': 'a8'}

dict_codes={'c1': 'Westside', 'c2': 'Zara', 'c3': 'Marks and spencers', 'c4': 'US Polo', 'c5': 'Armani', 'c6': 'Benetton', 'c7': 'Splash', 'c8': 'FCUK', 'f1': 'Wipro ', 'f2': 'Ikea', 'f3': 'Hulsta', 'f4': 'Style Spa', 'f5': 'Nilkamal', 'f6': 'Damro', 'f7': 'Durian', 'f8': 'Usha', 'p1': 'Montblanc', 'p2': 'Parker', 'p3': 'Linc', 'p4': 'Flair', 'p5': 'Reynolds', 'p6': 'Cello', 'p7': 'Camlin', 'p8': 'Classmate', 's1': 'Puma', 's2': 'Nike', 's3': 'Reebok', 's4': 'Asics', 's5': 'Adidas', 's6': 'Head', 's7': 'Wilson', 's8': 'Skechers', 'b1': 'Loreal', 'b2': 'Unilever', 'b3': 'Estee Lauder', 'b4': 'Coty', 'b5': 'Shiseido', 'b6': 'Beiersdorf', 'b7': 'Jhonson and Jhonson', 'b8': 'Amore Pacific', 'w1': 'Emporio Armani', 'w2': 'Fozzil', 'w3': 'Timex', 'w4': 'Titan', 'w5': 'Rado', 'w6': 'Tag Huer', 'w7': 'Casio', 'w8': 'Fastrack', 'm1': 'One Plus', 'm2': 'Apple', 'm3': 'Samsung', 'm4': 'Poco', 'm5': 'Redmi', 'm6': 'Huawei', 'm7': 'Oppo', 'm8': 'Vivo', 'l1': 'System76', 'l2': 'Toshiba', 'l3': 'Razor Blade', 'l4': 'Fujitsu', 'l5': 'HP', 'l6': 'Lenovo', 'l7': 'Alienware', 'l8': 'Asus', 'k1': 'Sleek', 'k2': 'Jhonson Kitchens', 'k3': 'Hafele', 'k4': 'Haecker', 'k5': 'Kohler', 'k6': 'ECBO', 'k7': 'Godrej Interio', 'k8': 'Style Spa', 'a1': 'Auto international', 'a2': 'Ayushi Engineering Company', 'a3': 'Arjan Industries', 'a4': 'Best Forgings India', 'a5': 'Bomrah Industries', 'a6': 'Chunho Tech', 'a7': 'Canara Standard', 'a8': 'Cyner Industrial'}


#------------------------------------------------------------
import copy

import ast
from ast import literal_eval

# df = pd.DataFrame(columns=['data'])
arr = []
app = Flask(__name__)
CORS(app)
df_yotube=pd.read_csv("prod_names_data.csv")#
df = pd.read_csv("carts2_new2.csv")
real_df_friends=pd.read_csv('friends_prod.csv')
# instances=np.array(df)
inputs = np.array(df.iloc[:, :-1])
outs = np.array(df.iloc[:, -1])
X_train, X_test, y_train, y_test = train_test_split(inputs, outs, test_size=0.20)

svclassifier = SVC(kernel='linear',probability=True)
svclassifier.fit(X_train, y_train)
# y_pred = svclassifier.predict(X_test)
# print(confusion_matrix(y_test,y_pred))

df_user_cart=pd.read_csv("user_history.csv")
df_product=pd.read_csv("prod_info.csv") 

def get_weights(cart_instance,data_frame):
    weights=[]
    for one_onehot in cart_instance:
        for i in range(len(data_frame)):
            if list(data_frame.loc[i])[0] == one_onehot:
            # print ("yayy")
                weights.append(list(data_frame.loc[i])[1:4])
    # print (weights)
    for i in weights:
        if i[1]=="low":
            i[1]=1
        elif i[1]=="medium":
            i[1]=2
        elif i[1]=="high":
            i[1]=3
    metric=[]
    for variable in weights:
        metric.append(variable[2]/variable[1])
    # print (metric)
    return metric    
def make_one_hot(cart_list,df_prodinfo):
    nof=len(df_prodinfo)
    cart_hot=[0]*nof
    # cart_one_hot=[cart_hot]
    prod_ids=list(df_prodinfo.iloc[:,0])
    for i in cart_list:
        cart_hot[prod_ids.index(i)]+=1
    return [cart_hot]


def recommendation(cart_list,svclassifier_input,df):
    # print (df)
    # prod_info=(df.iloc[:,:])
    prod_ids=list(df.iloc[:,0])
    # cart_one_hot=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    cart_one_hot=make_one_hot(cart_list,df)
    # cart_infos=['c1', 'c2', 'c3', 'c4', 'c5', 'f1', 'f2', 'f3', 'f4', 'f5', 'p1', 'p2', 'p3', 'p4', 'p5', 's1', 's2', 's3', 's4', 's5', 'b1', 'b2', 'b3', 'b4', 'b5', 'w1', 'w2', 'w3', 'w4', 'w5', 'm1', 'm2', 'm3', 'm4', 'm5', 'l1', 'l2', 'l3', 'l4', 'l5', 'k1', 'k2', 'k3', 'k4', 'k5', 'a1', 'a2', 'a3', 'a4', 'a5']
    # for product in cart_list:
    #     cart_one_hot[0][cart_infos.index(product)]+=1
    onecopy=[]
    # mine=[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    outp = svclassifier_input.predict(cart_one_hot)                                              #
    gonna_ex=[]
    out_class=outp[0]
    # probabilite=outprob[out_class]
    best_buy=[]
    if out_class==0:
        # print (cart_one_hot)
        ratios=get_weights(cart_list,df)
        # print (ratios)
        # final=sum(ratios)
        # if (final/len(cart_list)<2):
        x_org=prod_ids.index(cart_list[ratios.index(min(ratios))])
        # print (prod_info)
        gonna_ex=cart_list[ratios.index(min(ratios))]
        class_req=df.loc[x_org][1]
        # print (class_req)
        same_prods=[]
        for i in range(len(df)):
            
            is_it=list(df.loc[i])
            if is_it[1]==class_req:
                # print (i)
                same_prods.append(is_it)
        # print (same_prods)
        
        onecopy=copy.deepcopy(cart_list)
        ex=ratios.index(min(ratios))
        # print (ex)

        # print (onecopy)
        # onecopy[x_org]=0
        # x=(int(x_org/5))*5
        # print (x)
        
        for i in same_prods:
            onecopy[ex]=copy.deepcopy(i[0])
            
            # print (onecopy)
            created_list=copy.deepcopy(make_one_hot(onecopy,df))
            outp = svclassifier_input.predict(created_list)
            outprob = svclassifier_input.predict_proba(created_list)
            # print (outp)
            if outp[0]==1:
                # print ("it happened")
                best_buy.append([i[0],outprob[0][1]])
            # onecopy[i]=0
        print ("#########################",gonna_ex,best_buy)
        return gonna_ex,best_buy
    else:
        return 0            

def personalized(recommender_list,df,user_id):
    exc=recommender_list[0]
    recommender_list=copy.deepcopy(recommender_list[1])
    for j in recommender_list:
        j[1]*=10
    # print (recommender_list)
    for i in range(len(df)):
        converted=literal_eval(list(df.loc[i])[0])
        if converted[0]==user_id:
            # print (converted)
            count=[0]*len(recommender_list)
            for prod in range(len(recommender_list)):
                ide=recommender_list[prod][0]
                # print (ide)
                for instances in converted[1:]:
                    # print (instances)
                    if instances[0]=='1' or instances[0]==1:
                        # for ids in instances
                        try:
                            
                            instances.index(ide)
                            # print ('yayy')
                            recommender_list[prod][1]+=1
                            count[prod]+=1
                        except:
                            pass    
    # print (count)
    print (recommender_list)
    prob=0
    index_best=0
    for fi in range(len(recommender_list)):
        if prob<recommender_list[fi][1]:
            prob=recommender_list[fi][1]
            index_best=fi
    first=recommender_list.pop(index_best)
    prob=0
    index_best=0
    for fi in range(len(recommender_list)):
        if prob<recommender_list[fi][1]:
            prob=recommender_list[fi][1]
            index_best=fi
    second=recommender_list.pop(index_best)
    return {exc:[first[0],second[0]]} 

#------------------------------------------------

cart_global='System76'
link="https://www.youtube.com/watch?v=8DULIfYDyBo"
#------------------------------------------------
def getting_freinds(dic_recommendation,df_friends,present_user,df_prodinfo):
    rec_products=list(dic_recommendation.values())[0]
    final=[]
    for i in range(len(df_friends)):
        one=literal_eval(list(df_friends.loc[i])[0])
        if one[0]==present_user:
            for prod in rec_products:    
                product=[prod]
                req_ind=list(df_prodinfo.iloc[:,0]).index(prod)
                product.append(one[req_ind+1])
                final.append(product)
    return {list(dic_recommendation.keys())[0]:final}

@app.route('/')
def hello():
    return jsonify({'text': 'Hello World!'})

@app.route('/cart',methods=['POST','GET'])
def recv_data():
    global cart_global
    data=request.get_json(force=True)
    print("........",data)
    userid=list(data.keys())[0]
    print (userid)
    cart=data[userid]
    print (cart)
    new_cart=copy.deepcopy(cart)
    #---------------------
    for names in range(len(new_cart)):
        cart[names]=dict_names[new_cart[names]]
    #---------------------
    print ("----------------------")
    print (cart)
    cart_global=copy.deepcopy(cart) #--------------------------------
    multi_recomm=recommendation(cart,svclassifier,df_product)
    print (multi_recomm)
    if multi_recomm==0:
        # cart_global.pop()
        cart_global=(copy.deepcopy(new_cart[0]))
        sending_data={}
        print(sending_data)
        return(sending_data)
    else:
        sending_data=(personalized(multi_recomm,df_user_cart,userid))
        tochange= (getting_freinds(sending_data,real_df_friends,userid,df_product))
        print (tochange)
        enco_key=list(tochange.keys())[0]
        denko_key=dict_codes[enco_key]
        # cart_global.pop()
        cart_global=((denko_key))
        print ('-----------',cart_global)
        lists_of_lists=tochange[enco_key]
        for li in lists_of_lists:
            li[0]=dict_codes[li[0]]
        now_final={denko_key:lists_of_lists}

        print (now_final)
        # print(sending_data)
        return(now_final)
     
@app.route('/mouse', methods=['POST'])
def recv_mouse():
    # global cart_global
    data = request.get_json()
    # if(data == {}):
    #      df['data'] = arr
    #      df.to_csv("digital_behavior.csv", index=False)
    #      return data
    # arr.append(data)
#  print(data)

    #---------------------
    zipped=split_it(data)
    # inputs=get_x_y()
    #print (inputs)
    # print ("inputs_len",len(inputs))
    X=make_slopes(zipped)
#  print (X)
#  print ("slopes_len",len(X))
    new_X=make_3d(X)
    # print 
#  print("Newwwwwww",new_X)
    # load model from single file
    # model = load_model('lstm_model.h5')
    # make predictions
#  print ("ssssssssssssss")
    with graph.as_default():
        yhat = model.predict([new_X], verbose=0)
    print("ans:",yhat)
    noww=list(yhat[0])
    i =noww.index(max(noww))
    print ('#',cart_global)
    #---------------------

    # print (type(data))
    # print (predict(data,svclassifier))
    # print(arr)

    #---------------------
    if i==1:
        for_youtube=cart_global
        print (for_youtube)
        for iter in range(len(df_yotube)):
            inter=list(df_yotube.loc[iter])
            if inter[1]==for_youtube:
                link=inter[2]
                print ("===",for_youtube)
                return jsonify({link:i})
    else:
        return jsonify({"ans":i})


if __name__ == '__main__':
     app.run(port=5000)
