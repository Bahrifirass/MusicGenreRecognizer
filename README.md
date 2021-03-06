# MusicGenreRecognizer
Music Genre Recognition is an important field of research in Music Information Retrieval (MIR)
Music Genre Recognition involves making use of features such as spectrograms, MFCC’s for predicting the genre of music.
I am going to make use of GTZAN Dataset which is really famous in Music Information Retrieval (MIR). The Dataset comprises 10 genres 
Blues, Classical, Country, Disco, Hip Hop, Jazz, Metal, Pop, Reggae, Rock.
Each genre comprises 100 audio files (.wav) of 30 seconds each that means 800 training samples vs 200 validation in case of 80% and 20% Split
i've already tried only this 1000 audio samples and the results wasn't really good (>0.2)
that's why i decided to split each file into 10 files of 3 seconds which gave me 1000 samples per genre and 10000 in total as well as
data augmentation with the famous keras's ImageDataGenrator, i've got a 76% accuracy
Note: there was some errors in generating spectograms from the jazz genre, so i had only the half of samples(440 samples)

