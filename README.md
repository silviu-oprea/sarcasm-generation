# Max: An Explainable Sarcastic Response Generator

This repository contains the implementation of the Max sarcasm generator, mentioned in the following published papers:
* Silviu Vlad Oprea, Steven Wilson, and Walid Magdy. 2021. Chandler: An Explainable Sarcastic Response Generator. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 339–349. Association for Computational Linguistics. [link](https://aclanthology.org/2021.emnlp-demo.38/).
* Silviu Vlad Oprea, Steven Wilson, and Walid Magdy. 2022. Should a Chatbot be Sarcastic? Understanding User Preferences Towards Sarcasm Generation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7686–7700. Association for Computational Linguistics. [link](https://aclanthology.org/2022.acl-long.530/).

Bibtex keys for these papers are listed below.

## Background

Previous sarcasm generators assume the intended meaning that sarcasm conceals is the opposite of the literal meaning. We argue that this traditional theory of sarcasm provides a grounding that is neither necessary, nor sufficient, for sarcasm to occur. Instead, we ground our generation process on a formal theory that specifies conditions that unambiguously differentiate sarcasm from non-sarcasm. Furthermore, Max not only generates sarcastic responses, but also explanations for why each response is sarcastic. This provides accountability, crucial for avoiding miscommunication between humans and conversational agents, particularly considering that sarcastic communication can be offensive. In human evaluation, Max achieves comparable or higher sarcasm scores, compared to state-of-the-art generators, while generating more diverse responses, that are more specific and more coherent to the input.

Please see [this video](https://aclanthology.org/2021.emnlp-demo.38.mp4) that introduces Max (first paper above; Max named Chandler in the video)
and also [this video](https://aclanthology.org/2022.acl-long.530.mp4) that shows a social study on how users react to sarcastic responses generated by Max (second paper above).

## Running Max

Please note the code base is still not fully mature, but it should work if you follow the steps below.

### Environment setup

First, install the dependencies (as listed below; or using the package manager of your choice; or the enclosed `requirements.txt`):

```bash
pip install torch==2.2.2 ftfy==5.1 spacy==3.8.2 pandas==2.2.3 lemminflect==0.2.3 transformers==4.45.2 scipy==1.14.1 ipython==8.28.0
python -m spacy download en_core_web_sm
```

First, clone this repository:

```bash
git clone git@github.com:silviu-oprea/sarcasm-generation.git
```

Max uses COMET ([Bosselut et al., 2019](https://aclanthology.org/P19-1470/)), a language model fine-tuned to complete commonsense relations. Fetch the COMET codebase as follows:

```bash
cd sarcasm-generation/src/max/commonsense_builders/
git clone https://github.com/atcbosselut/comet-commonsense.git comet
```

Next, run the following scripts to setup COMET:

```bash
# Navigate back to the root of the project
cd comet
bash scripts/setup/get_atomic_data.sh
bash scripts/setup/get_model_files.sh
```

Next, download COMET pretrained models from `https://drive.google.com/open?id=1FccEsYPUHnjzmX-Y5vjCBeyRt1pLo8FB`. That is an archive called `pretrained_models.tar.gz`. Once downloaded, extract it. The result will be a folder `pretrained_models`. Move that folder to `src/max/commonsense_builders/comet/`.

Finally, run the followin script to finish setting up COMET:

```bash
python scripts/data/make_atomic_data_loader.py
```

Finally, move back to the root directory (where you cloned the sarcasm-generation repo). Therein, copy the data and model folders from COMET:

```bash
cp -r src/max/commonsense_builders/comet/data .
cp -r src/max/commonsense_builders/comet/model .
```

### Sarcastic response generation

First, create a file that contains a list of prompts to which sarcasmtic responses should be generated, one prompt per line.
Say that file is stored at `input/events.txt`, and you would like the responses, along with associated metadata (see papers above) to be stored at `output/responses.jsonl`.
To generate those responses, use:

```bash
python src/main.py --event_file_path input/events.txt --output_file_path output/responses.json
```

Here is a sample output for the prompt "I ran out of characters":

```json
{
  "event": "I ran out of characters",
  "failed_expectation": "I did not run out of characters",
  "relation_type": "xNeed",
  "relation_subject": "event",
  "relation_object": "know how to write",
  "norm_violated": "maxim of quality",
  "response_texts": [
    "Brilliant! Well done not knowing how to write.",
    "You didn't know how to write, that's for sure. Well done!"
  ]
}
```

Please consult the papers and/or for the meaning of each field.

### Citation

I hope you found this fun and instructive.
If you are using this codebase, please consider citing our papers:

```latex
@inproceedings{oprea-etal-2021-chandler,
    title = "Chandler: An Explainable Sarcastic Response Generator",
    author = "Oprea, Silviu  and
      Wilson, Steven  and
      Magdy, Walid",
    editor = "Adel, Heike  and
      Shi, Shuming",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-demo.38",
    doi = "10.18653/v1/2021.emnlp-demo.38",
    pages = "339--349"
}
```

and

```latex
@inproceedings{oprea-etal-2022-chatbot,
    title = "Should a Chatbot be Sarcastic? Understanding User Preferences Towards Sarcasm Generation",
    author = "Oprea, Silviu Vlad  and
      Wilson, Steven  and
      Magdy, Walid",
    editor = "Muresan, Smaranda  and
      Nakov, Preslav  and
      Villavicencio, Aline",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.530",
    doi = "10.18653/v1/2022.acl-long.530",
    pages = "7686--7700"
}
```

Thank you 🙂!
