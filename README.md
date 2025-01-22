This is a package to post-process your static creep experimental data.

It was first introduced in the paper "Experimental Challenges in Static Creep Testing of Cement Pastes".

The code can be adapted to treat data from virtually any data-acquisition system.

The original code was developed to treat data acquired by data-acquisition systems manufactured by the company INEGI, and by a custom-made LVDT multiplexer based on National Instruments modules.

But if you are using a different data-acquisition system, the code can still be useful to you. You would just have to implement proper data reading in the method **Experiment.readCreep_Batch** to match the formatting of your result file. Once you handle that and make sure you read your test data an dput it in the same format as the original method **Experiment.readINEGI_Batch**, the rest of the code can be promptly used. You may consult the methods **Experiment.readINEGI_Batch** and **Experiment.readNational_Batch** to see how such process differs for two different data-acquisition systems.
