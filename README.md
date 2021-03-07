# MusicGenreRecognizer
“Music gives a soul to the universe, wings to the mind, flight to the imagination, and life to everything.” – Plato
So through my new journey into the AI music research, I’m trying to break the struggles working on various projects and this was the first, an implementation of a CNN model for music recognition. 
Music Genre Recognition is an important field of research in Music Information Retrieval (MIR)
Music Genre Recognition involves using features such as spectrograms, MFCC’s.. for predicting the genre of music.
I am going to use the GTZAN Dataset which is really famous in MIR. The Dataset contains 10 genres 
Blues, Classical, Country, Disco, Hip Hop, Jazz, Metal, Pop, Reggae, Rock.
Each genre comprises 100 audio files (.wav) of 30 seconds each that means 800 training samples vs 200 validation in case of 80% and 20% Split
i've already tried only this 1000 audio samples and the results wasn't really good (>0.2)
that's why i decided to split each file into 10 files of 3 seconds which gave me 1000 samples per genre and 10000 in total as well as
data augmentation with the famous keras's ImageDataGenrator, i've got a 76% accuracy
Note: there was some errors in generating spectograms from the jazz genre, so i had only the half of samples(440 samples)
Furthermore, i've tested the model on some samples which i included two of them on the repository:
{'blues': 0,
 'classical': 1,
 'country': 2,
 'disco': 3,
 'hiphop': 4,
 'jazz': 5,
 'metal': 6,
 'pop': 7,
 'reggae': 8,
 'rock': 9}
1) sampletest: short solo of a distortion guitar sound Sample [[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]] (metal)
2) sampletest2 : short smooth piano loop [[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]](classical)


