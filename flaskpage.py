from flask import Flask,request,redirect, render_template, url_for, flash
import pandas as pd
import pickle
app = Flask(__name__)

filename = 'DiabetesKnn.pkl'
knn_model = pickle.load(open(filename, 'rb'))

#defining column data headers for input as a tuple(not used)
datacolumns = ("pregnancies",               "Glucose",                              
               "Blood Pressure",            "Skin Thickness",     
               "Insulin",                    "BMI",                
               "DiabetesPedigreeFunction",   "Age",
               "outcome",  
               )



@app.route('/Home_Page')
def homepage():
   return render_template('index.html')

@app.route('/')
def home():

   return render_template('index.html')


@app.route('/formm',methods=["POST","GET"])
def formm():
   if request.method == "POST":
      data1 = request.form.get('Editbox1')
      data2 = request.form.get('Editbox2')
      data3 = request.form.get('Editbox3')
      data4 = request.form.get('Editbox4')
      data5 = request.form.get('Editbox5')
      data6 = request.form.get('Editbox6')
      data7 = request.form.get('Editbox7')
      data8 = request.form.get('Editbox8')
      #print(data1,data2,data3,data4,data5,data6,data7,data8)
      row_df = pd.DataFrame([pd.Series([data2,data5,data6,data8])])
      #print(row_df)
      prediction=knn_model.predict_proba(row_df)
      output='{0:.{1}f}'.format(prediction[0][1], 2)

      output = str(float(output)*100)+'%'
      #print(output)

      if output>str(0.5):
         #print('iffff')
         prediction=f'The Probability of Occurence is {output} \n (Higher chance of having diabetes)'
         #print(prediction)
         
         return render_template('result.html',prediction=prediction )
         
      else:
         
         prediction=f'Probability of Occurence is {output} \n (lower chance of having diabetes) '
         #print(prediction)
         return render_template('result.html',prediction=prediction)
  


if __name__ == '__main__':
   app.run(debug=True)