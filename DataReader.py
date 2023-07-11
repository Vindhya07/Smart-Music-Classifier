import re

#function to read one folder to 2d array
def readData(path,emotion):
    Data=[]
    lyrics=[]
    for i in range(1,107):
        filename=path+emotion+"_"+str(i)+".txt"
        try:
            with open(filename,'r',encoding='latin-1') as file:
                lyric=file.read()
                (lyric,count)=re.subn(r"(\n|\t)"," ",lyric)
                (lyric,count)=re.subn(r"(\[\d\d:\d\d\.\d\d\])","",lyric)
                #(lyric,count)=re.subn(r"(\(.\)\)","",lyric)
                lyrics.append(lyric)
                # print("\n\nThese are the lyrics from datareadrer:\n")
                # print(lyrics)
        except:
            break


    #with codecs.open(path+"info.txt", mode='r', errors='ignore') as file:
    with open(path+"info.txt",'r',encoding='latin-1') as file:
        i=0
        for line in file:
            # print("\n\nPRINTING THE LINE\n")
            # print(line)
            row=line.split(':')
            row=[i.strip() for i in row]
            #row.append(emotion)
            # print("\n\nPRINTING row\n")
            # print(row)
            # print("\n\nPRINTING LYRICS[I]\n\n")
            # print(lyrics[i])
            row.append(lyrics[i])
            Data.append(row)
            ###print("\n\nPrintinf the appended data!!!!\n\n\n")
            ##print(Data)
            ##print("\n\nshowing i\n");
            # print(i);
            i+=1

    
    return Data

