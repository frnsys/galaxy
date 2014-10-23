function reset_pickles {
    rm concept_pipeline.pickle
    rm /tmp/*.pickle
}

echo -e "\nhandpicked.json\n\n"

echo -e "No feature reduction\n"
python run.py evaluate eval/data/event/handpicked.json
reset_pickles

echo -e "\n100 components\n"
python run.py train_concepts 100
python run.py compare eval/data/event/handpicked.json
reset_pickles

echo -e "\n500 components\n"
python run.py train_concepts 500
python run.py compare eval/data/event/handpicked.json
reset_pickles
