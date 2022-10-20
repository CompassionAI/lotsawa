# CompassionAI Lotsawa - tools for translating and understanding classical Tibetan

This is a collection of end-user tools to help with translation and understanding of long texts in classical Tibetan, especially the Kangyur and Tengyur.

For example, in your terminal you can:

```bash
# Bring up an interactive tool for translating individual short Tibetan sections
lotsawa-translate

# Output a translation of the Heart Sutra into English under ./translations
lotsawa-translate mode=batch mode.input_glob=heart_sutra.bo

# After translating to English, re-translate the Heart Sutra into simplified Chinese
lotsawa-retranslate target_language_code=zho_Hans

# Bring up an interactive tool for splitting Tibetan sections into words and tagging those words as nouns/verbs/adjectives/etc
lotsawa-words
```

A sample set of Tibetan documents to experiment on is available at <https://compassionai.s3.amazonaws.com/public/translation_test_docs.zip>.

Lotsawa is backed by our novel models that are the results of a research program into how to convert existing state-of-the-art translation models for short sentences, such as NLLB (No Language Left Behind), into models that are better able to handle the ambiguity of long classical Tibetan texts. The models used in Lotsawa utilize pre-trained, state-of-the-art translation models as a backbone that have had their neural architectures significantly modified to accomodate long texts. In particular, we are not simply serving (fine-tuned) NLLB; we are serving a model with a new neural architecture that's much better than NLLB at handling Tibetan ambiguity. Lotsawa implements a carefully tuned end-to-end translation pipeline for long texts - the result of many experiments on strategies for the preservation of contextual semantic information in the low-resource setting of classical Tibetan. Please see <https://www.compassion-ai.org/> for an explanation of our research.

> We are a tiny team of volunteers on a shoestring budget. The community of people who would benefit from these tools is likewise very small. If we don't work together, these tools will struggle to improve and be useful.
>
> **PLEASE** don't immediately give up and walk away if you run into a problem. Without at least a tiny bit of ***your*** help these tools will never evolve to benefit anyone. Please, for the sake of the Tibetan language and the Dharma, contact us before giving up.
>
> Contact us if you're using these tools, if something isn't working and you need help, if the tools are performing poorly, just to say hi, or for any other reason.
>
> We can be reached at contact@compassion-ai.org or on GitHub issues.

## Installation

We assume you're on a Mac. The installation should work on Windows and Linux _mutatis mutandis_.

### Basic instructions

Install with pip:

```bash
pip install lotsawa
```

Lotsawa requires Python 3.6 or greater. This shouldn't be a problem on almost any modern computer. If you are having issues with this on an older Mac, see the Homebrew documentation here: <https://docs.brew.sh/Homebrew-and-Python>. If you can't make it work or if the Homebrew docs are too much, contact us at <contact@compassion-ai.org> or open a GitHub issue.

### Basic instructions - NVidia GPUs

**This section does not apply to Macs. Newer Macs with M1 chips or better should use the embedded GPU by default.**

If you have an NVidia GPU and want to use it to massively speed everything up - we strongly recommend doing this if you can - you will need to install CUDA-enabled PyTorch. Begin by installing the NVidia drivers and CUDA:

 - Windows: <https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local>.
 - Linux: There's lots of tutorials for the common distros, as well as the NVidia official document at <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>.
 - Cloud: start with an image with CUDA pre-installed, or follow the instructions for your specific OS.

Then install CUDA-enabled PyTorch. This is very easy. The following line in your terminal should work:

```bash
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116
```

If for some reason it doesn't, follow the instructions on <https://pytorch.org/get-started/locally/>:

 - Set the PyTorch build to Stable.
 - Set package to pip.
 - Set language to Python.
 - Set compute platform to CUDA 11.6 or greater.

It will give you a line of code to run, paste it into your terminal and you should be good. If you're not good, start without CUDA and contact us at <contact@compassion-ai.org> or open a GitHub issue.

As the usage of Lotsawa evolves we may simplify this process as needed.

### Slightly more advanced - conda

If you're up to it, we recommend using a virtual environment to simplify your installation and management of your installed software. In our experience, conda is the easiest way to do this. Conda will keep the stuff needed to run Lotsawa separate from the rest of your computer. This way, if anything breaks, you can easily uninstall Lotsawa without affecting the rest of your programs.

*Before installing with `pip`*, begin by installing miniconda from here: <https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html>, then run the following:

```bash
conda create -n lotsawa
conda activate lotsawa
conda install -c conda-forge python>=3.6 pip
pip install lotsawa
```

Whenever you want to use Lotsawa, activate your virtual environment:

```bash
conda activate lotsawa
lotsawa-translate   # Or whatever you want to do
```

When you're done, either just close the terminal window or run:

```bash
conda deactivate
```

To uninstall Lotsawa and all the associated packages, including PyTorch, all you need to do is:

```bash
# Delete the virtual environment, including Lotsawa itself
conda env remove -n lotsawa

# Clear the model cache
echo rm -rf $(python -c "from torch.hub import get_dir; print(get_dir() + '/champion_models')") | bash
```

### Developers - installing from source

**PLEASE** begin by dropping us a line at <contact@compassion-ai.org> so that we can work with you to make you successful. We have no sales team or anything like that, we will not hassle you, we just want to be helpful.

You will need to clone four CompassionAI repos: common, manas, garland and lotsawa:

```bash
export CAI_BASE_DIR=~/workspace/compassionai   # Or wherever you like
mkdir -p $CAI_BASE_DIR; cd $CAI_BASE_DIR
git clone git@github.com:CompassionAI/common.git
git clone git@github.com:CompassionAI/manas.git
git clone git@github.com:CompassionAI/garland.git
git clone git@github.com:CompassionAI/lotsawa.git
```

We strongly recommend using conda. In fact, we recommend mamba - it is much faster than conda, with no downside. We provide a minimal environment file for your convenience.

```bash
# Install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Install mamba
conda install mamba -c conda-forge

# Create the Lotsawa virtual environment
cd $CAI_BASE_DIR/lotsawa      # Or wherever you cloned the repo
mamba env create -f env.yml
conda activate lotsawa
```

### Research

**PLEASE** begin by dropping us a line at <contact@compassion-ai.org> so that we can work with you to make you successful. We have no sales team or anything like that, we will not hassle you, we just want to be helpful.

We *very strongly* recommend doing research *only* on Linux.

If you're planning to do research, i.e. tinker with our datasets or tweak the models, you probably want the data registry:

```bash
git@github.com:CompassionAI/data-registry.git
cd data-registry
./pull.sh   # Warning: large download
```

Follow the installation instructions in CompassionAI/common for research.

## Usage

### Translation into English

Use the `lotsawa-translate` utility. It has two modes: interactive and batch.

 - Interactive mode will prompt you for individual short Tibetan sections and will output English translations. This is intended as a test or a demo.
 - Batch mode will process long Tibetan files (in Unicode with uchen script). This mode will involve segmentation of the long text into shorter sections, followed by sequential translating with context. NB: the segmented sections will not translate the same in batch as in interactive mode due to the use of context during translation.

Interactive is the default mode. An example that uses batch mode is:

```bash
lotsawa-translate mode=batch mode.input_glob=~/tibetan_texts/*.bo
```

This will translate all texts in the directory `~/tibetan_texts` that have the extension `.bo` and output the results to `./translations`. To control the output directory, set `mode.output_dir`.

A sample set of Tibetan documents to experiment on is available at <https://compassionai.s3.amazonaws.com/public/translation_test_docs.zip>.

To use CUDA, pass in `cuda=true`. For example:

```bash
lotsawa-translate mode=batch mode.input_glob=~/tibetan_texts/*.bo cuda=true
```

If your GPU has less than 8GB of VRAM you may see CUDA OOM errors. We recommend reducing the number of beams during beam search. The easiest way to do this is as follows:

```bash
lotsawa-translate mode=batch mode.input_glob=~/tibetan_texts/*.bo cuda=true generation=slow     # 50 beams, default
lotsawa-translate mode=batch mode.input_glob=~/tibetan_texts/*.bo cuda=true generation=medium   # 20 beams
lotsawa-translate mode=batch mode.input_glob=~/tibetan_texts/*.bo cuda=true generation=fast     # 5 beams
```

We recommend trying `cuda=false generation=slow` on some sample text to compare against. If you're unhappy with the results and would benefit from a more complex memory management protocol during beam decoding, please contact us at <contact@compassion-ai.org> or open a GitHub issue.

See the full help for `lotsawa-translate` for a complete list of options:

```bash
lotsawa-translate --help
```

Advanced options:

 - Use `bad-words-list` to create word exclusion lists during translations.
 - You can provide configuration overrides on a per-file basis. In the same folder as the Tibetan file you can provide a YAML configuration file with overrides. For example, see the override file for the Manjusrinamasamgiti in the data registry under `processed_datasets/translation-test-docs`.

### Re-translation into other languages

To translate into languages other than English, we find that the best results were to translate to English first and then zero-shot translate from English to the target language using NLLB. We provide the simple tool `lotsawa-retranslate` to facilitate this. This strategy works best for translation into other high resource languages such as simplified Chinese.

If you are trying to use this tool but are still running into issues please contact us at <contact@compassion-ai.org> or on our GitHub issues page. Some issues you may face could be: seeing a lot of English in the target output, toxicity or other bad words, or excessive pronoun/context switching. While we saw quite good results with this tool, we are not professional translators. We are likely to be able to improve the model if we understand your use case, _please_ contact us.

The tool will translate all English files that match the input glob into the target language. The input glob defaults to `translations/*.en` and the output extension defaults to the language code. For a readable list of the language codes, please see table 1, _204 Languages of No Language Left Behind_, on pages 13-16 in the NLLB model paper at <https://arxiv.org/pdf/2207.04672>. You can also use the argument `list_language_codes=true` to print out all language codes.

Pass in `cuda=true` to use an NVidia GPU. You shouldn't run out of memory with the settings used here. If you are, please [contact us](contact@compassion-ai.org).

As an example, to translate a directory with Tibetan texts in it into simplified Chinese:

```bash
# First, translate into English
lotsawa-translate mode=batch mode.input_glob=~/tibetan_texts/*.bo

# Reviewing the English translation here will improve the Chinese

# Finally, re-translate the English into Chinese
lotsawa-retranslate target_language_code=zho_Hans
```

The results will be in `./translations` with the extension `.zho_Hans`.

You do not need to use `lotsawa-translate` to produce the English text. The `lotsawa-retranslate` tool will go through the input files line by line, skip any lines with Tibetan characters in them, and translate each remaining line using NLLB. The most important thing to know is: NLLB works well on _short_ inputs. A simple approach with English would be to split every English sentence into its own line. **PLEASE** contact us at <contact@compassion-ai.org> so that we can help, or open an issue on our GitHub page.

If you're interested in using the 84,000 XML files, note that the tool will not do any preprocessing, such as unfolding the XML tags. The class `TeiLoader`, found in the CompassionAI/common repo under `cai_common/data/tei_loader.py`, uses BeautifulSoup to extract and clean the translations from the 84,000 XML files. _Please_ contact us if you're interested in using this code.

See the full help for `lotsawa-retranslate` for a complete list of options:

```bash
lotsawa-retranslate --help
```

### Word segmentation and part-of-speech tagging

Currently we provide only an interactive tool for this to help you assess the performance for your needs and as an example for how to use our Python packages. If you have a use case for our token classifiers that needs a different delivery of the models, or if you need us to change how the models themselves work, _please_ contact us at <contact@compassion-ai.org> or open an issue on our GitHub page.

To run the tool, just activate your conda environment (if any) and use:

```bash
lotsawa-words
```

The tool currently has no user-configurable options. We expect to eventually update the models underlying this tool, especially the tokenization.

### Cleaning the model cache

Lotsawa will download the trained CompassionAI language models into the PyTorch Hub cache. The models can get fairly large, for example our current best model for Tibetan-English translation is 1.8GB.

To clear the cache, simply delete the PyTorch Hub cache. This is safe, if you run Lotsawa again it will re-create the cache and re-download the models. To find the cache directory, run this in your terminal:

```bash
python -c "from torch.hub import get_dir; print(get_dir())"
```

This will print the directory, which you can then explore and delete if you like. To delete only the CompassionAI cache with a single terminal command, use:

```bash
echo rm -rf $(python -c "from torch.hub import get_dir; print(get_dir() + '/champion_models')") | bash
```

## For developers

### Hydra

Lotsawa uses Hydra for its configuration. Hydra is a tool developed by Facebook to manage complex configuration, especially in the machine learning space. It enables reproducible results, configuration as code, grouping of configuration options, and easy defaults and overrides. See <https://hydra.cc/> for details.

### Embedding Lotsawa's backend into your own applications

The utilities we provide in the Lotsawa package are simple wrappers around helper classes provided by the Garland and Manas CompassionAI repos. The helper classes encapsulate the loading of the models, the encoding and decoding of the text, and implement any algorithms we layered on top of the model decoding to create our results. For example, the `Translator` class encapsulates the process of maintaing the target language context during batch translation.

The source code for the provided utilities can be found in:

```bash
lotsawa/lotsawa/translate.py
lotsawa/lotsawa/retranslate.py
lotsawa/lotsawa/part_of_speech.py
```

### Fine-tuning Lotsawa's models

We provide the code for this in the Garland and Manas repositories. Documentation may be sparse currently. _Please_ contact us at <contact@compassion-ai.org> or open an issue on our GitHub page. We have no sales team or anything like that, we will not hassle you, we just want to be helpful.