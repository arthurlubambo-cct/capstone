MTG-QBH: dataset of a cappella sung melodies 
============================================

Created By
----------

Justin Salamon*, Joan Serra^ and Emilia Gomez*
* Music Technology Group, Universitat Pompeu Fabra, Barcelona, Spain
^ Artificial Intelligence Research Institute (IIIA-CSIC), Spanish National Research Council, Bellaterra, Spain 
http://www.justinsalamon.com
http://mtg.upf.edu/


Description
-----------

This dataset includes 118 recordings of sung melodies. The recordings were made as part of the experiments on 
Query-by-Humming (QBH) reported in the following article:

J. Salamon, J. Serra and E. Gomez, "Tonal Representations for Music Retrieval: From Version Identification to 
Query-by-Humming", International Journal of Multimedia Information Retrieval, special issue on Hybrid Music Information Retrieval, In Press (accepted Nov. 2012) 

The recordings were made by 17 different subjects, 9 female and 8 male, whose musical experience ranged from 
none at all to amateur musicians. Subjects were presented with a list of songs out of which they were asked to 
select the ones they knew and sing part of the melody. The subjects were aware that the recordings will be used as 
queries in an experiment on QBH. There was no restriction as to how much of the melody should be sung nor which 
part of the melody should be sung, and the subjects were allowed to sing the melody with or without lyrics. The 
subjects did not listen to the original songs before recording the queries, and the recordings were all sung 
a capella without any accompaniment nor reference tone. To simulate a realistic QBH scenario, all recordings 
were done using a basic laptop microphone and no post-processing was applied. The duration of the recordings 
ranges from 11 to 98 seconds, with an average recording length of 26.8 seconds. 

In addition to the query recordings, three meta-data files are included, one describing the queries and two 
describing the music collections against which the queries were tested in the experiments described in the 
aforementioned article. Whilst the query recordings are included in this dataset, audio files for the music 
collections listed in the meta-data files are NOT included in this dataset, as they are protected by copyright 
law. If you wish to reproduce the experiments reported in the aforementioned paper, it is up to you to obtain 
the original audio files of these songs.

All subjects have given their explicit approval for this dataset to be made public.   


Audio Files Included
--------------------

118 audio files of sung melodies (see description above) in 16 bit mono WAV format sampled at 44.1kHz 


Meta-data Files Included
------------------------

Collection_Full.csv

This file contains meta-data information about the full collection of songs against which the queries were 
evaluated in the experiments described in the aforementioned article. The collection was compiled in the context of:

J. Serra, "Identification of versions of the same musical composition by processing audio descriptions", PhD Thesis, 
Universitat Pompeu Fabra, Barcelona, Spain.

In total 2125 recordings are described, and for each recording there are six columns:

- Song ID: a unique and arbitrary identifier given to each recording. NOTE: the ID is unique for every 
recording, meaning different versions (recordings) of the same song will have different IDs.
- Title: the title of the song. Note that recordings of the same song may have different titles, for example
if they are sung in different languages.
- Artist: the performing artist for this specific recording.
- Original Artist: the composer of the song.
- Canonical: whether this recording is the canonical version of this song. If it is, the field is set to 'YES'. 
Otherwise (i.e. not the canonical version) the field is left empty. NOTE: this labeling was done by the authors 
of this dataset and is completely subjective.
- Class label: a unique label for all recordings of the same musical piece. This is the only field guaranteed
to be identical for all recordings which are considered by the authors of the dataset to be versions of the same 
musical piece. 


Collection_Canonicals.csv

This file contains meta-data information about the collection of canonical songs against which the queries were 
evaluated in the experiments described in the aforementioned article. Unlike the full collection, this collection
includes only one version of every musical piece, the one considered by the authors of the dataset to be the 
canonical version of the piece. This means this collection is a SUBSET of the full collection above. 
NOTE: the selection of canonical songs was done by the authors of the dataset and is completely subjective. 
In total 481 recordings are described, and for each recording there are six columns, as for the full collection. 
Since this collection is a subset of the full collection, the column values for a recording with certain Song ID
in Collection_Canonicals.csv will be identical to the column values for the same recording (i.e. same Song ID) 
in Collection_Full.csv.


Queries.csv

This file contains meta-data information for the query recordings. In total 118 recordings are described, and
for each recording there are six columns:

- Filename: the name of the audio file.
- Query ID: the ID of the query.
- Song ID: the ID of the song to which the query corresponds. NOTE: use this ID to match the query to the song
to which it corresponds in Collection_Full.csv and Collection_Canonicals.csv.
- Original Artist: the composer of the song to which the query corresponds.
- Class label: a unique label for all recordings of the same musical piece. This is the only field guaranteed
to be identical for all recordings (whether query or song) which are considered by the authors of the dataset
to be versions of the same musical piece.


Please Acknowledge MTG-QBH in Academic Research
-----------------------------------------------

When MTG-QBH is used for academic research, we would highly appreciate if scientific publications of works partly 
based on the MTG-QBH dataset cite the following publication:

J. Salamon, J. Serra and E. Gomez, "Tonal Representations for Music Retrieval: From Version Identification to 
Query-by-Humming", International Journal of Multimedia Information Retrieval, special issue on Hybrid Music Information Retrieval, In Press (accepted Nov. 2012)

The compilation of this dataset was supported by: Programa de Formacion del Profesorado Universitario (FPU) of the Ministerio de Educacion de Espana, Consejo Superior de Investigaciones Cientificas (JAEDOC069/2010), Generalitat de Catalunya (2009-SGR-1434), the European Commission, FP7 (Seventh Framework Programme), MIReS project ICT-2011.1.5, grant agreement No. 287711 and TECNIO network promoted by ACC1Ó agency by the Catalan Government. 


Conditions of Use
-----------------

Dataset compiled and authored by Justin Salamon, Joan Serra and Emilia Gomez. Copyright © 2012 Music Technology Group, Universitat Pompeu Fabra. All Rights Reserved.
 
The MTG-QBH dataset is offered free of charge for internal non-commercial use only.  You may not redistribute, publically communicate or modify it, unless expressly permitted by the Universitat Pompeu Fabra (UPF) or by applicable law. The individual contents of the dataset may be protected by copyright, and a non-transferrable limited license to reproduce and use the same is granted for the indicated purpose only.
 
The dataset and its contents are made available on an "as is" basis and without warranties of any kind, including without limitation satisfactory quality and conformity, merchantability, fitness for a particular purpose, accuracy or completeness, or absence of errors. Subject to any liability that may not be excluded or limited by law, the UPF is not liable for, and expressly excludes, all liability for loss or damage however and whenever caused to anyone by any use of the MTG-QBH dataset or any part of it.


Feedback
--------

Problems, positive feedback, negative feedback... it is all welcome! Please help us improve MTG-QBH by sending your 
feedback to: justin.salamon@upf.edu AND mtg@upf.edu

In case of a problem report please include as many details as possible.
