from flask import Flask,render_template, request, redirect, url_for
from werkzeug import secure_filename
import getLy as gl
import NewPredictions_M as n_predict
import cdt
import models
import record
import cqt
import onset

app = Flask(__name__)

@app.route('/',methods = ['POST', 'GET'])
def index():
        return render_template('sample2.html')

@app.route('/genre1',methods = ['POST', 'GET'])
def record_song():
	if request.method == 'POST':
		flag=record.main()
		cdt.song_name = "output.wav"
		ouputText=models.pred()
		return render_template('genre1.html',data=ouputText.upper())
			 
	return render_template('genre1.html',data=" ")

@app.route('/genre2',methods = ['POST', 'GET'])
def genre2():
       return render_template('genre2.html')
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	 if request.method == 'POST':
		  f = request.files['file']
		  f.save(secure_filename(f.filename))
		  print("\n\nFIle name : ")
		  print(f.filename)
		  print("\n")
		  cdt.song_name=str(f.filename)
		  ouputText=models.pred()
		  #times = onset.detect_onset(f.filename)
		  #cqt.cqt(cqt.load(str(f.filename)))
		  return render_template('genre2.html',data=ouputText.upper())
	 return render_template('genre2.html',data=" ")

@app.route('/mood1',methods = ['POST', 'GET'])
def mood1():
        if request.method == 'POST':
             song_n = request.form['text']
             artist_n = request.form['text2']
             print ( request.form )
             gl.get_lyrics(song_n, artist_n)
             ouputText=n_predict.mainPredict()
             return render_template('mood1.html',data=ouputText)
        return render_template('mood1.html',data=" ")




@app.route('/mood2',methods = ['POST', 'GET'])
def mood2():
        return render_template('mood2.html')
@app.route('/uploader1', methods = ['GET', 'POST'])
def upload_file1():
             if request.method == 'POST':
                  f = request.files['file']
                  f.save(secure_filename(f.filename))
                  print("\n\nFIle name : ")
                  print(f.filename)
                  print("\n")
                  n_predict.filename=str(f.filename)
                  ouputText=n_predict.mainPredict()
                  return render_template('mood2.html',data=ouputText)
             return render_template('mood2.html',data=" ")
@app.route('/sample2',methods = ['POST', 'GET'])
def sample2():
    return render_template('sample2.html')




if __name__ == '__main__':
    app.run(debug=True)
