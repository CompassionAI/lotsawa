import os
import sys
import glob
import yaml
import logging
import unicodedata

import hydra
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm


@hydra.main(version_base="1.2", config_path="translation_config", config_name="retranslate")
def main(cfg):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    logging.info("Loading re-translation model")
    logging.debug("  Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_model_name, src_lang="eng_Latn")
    if cfg.output.list_language_codes:
        for lang_code in sorted(list(tokenizer.lang_code_to_id.keys())):
            print(lang_code)
        return
    logging.debug("  Loading model")
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.hf_model_name)
    if cfg.model.cuda:
        logging.debug("  Copying model to GPU")
        model.cuda()

    if cfg.generation.word_exclusion_list is not None:
        if not os.path.isfile(cfg.generation.word_exclusion_list) or \
            not os.path.splitext(cfg.generation.word_exclusion_list)[1] in {'.yaml', '.yml'} \
        :
            logging.error("The word exclusion list should be a YAML file")
            raise FileNotFoundError("The word exclusion list should be a YAML file")
        logging.info("Tokenizing word exclusion list")
        with open(cfg.generation.word_exclusion_list, 'r') as f:
            bad_words = yaml.safe_load(f)
        bad_words = bad_words.get(cfg.output.target_language_code, [])
        bad_word_tokens = [tokenizer.encode(w, add_special_tokens=False) for w in tqdm(bad_words)]
    else:
        bad_word_tokens = []

    logging.info("Re-translating")
    output_ext = getattr(cfg.output, "output_extension", cfg.output.target_language_code)
    in_fns = glob.glob(cfg.input_glob)
    for in_fn in (files_pbar := tqdm(in_fns)):
        files_pbar.set_description(os.path.basename(in_fn))
        with open(in_fn, 'r') as f_in:
            translated = [l.strip() for l in f_in.readlines()]
        out_fn = os.path.join(cfg.output.output_dir, os.path.splitext(os.path.basename(in_fn))[0] + '.' + output_ext)
        with open(out_fn, 'w') as f_out:
            for line in tqdm(translated, leave=False, desc="Translating"):
                if any('TIBETAN' in unicodedata.name(c) for c in line):
                    f_out.write(line + '\n')
                elif len(line) == 0:
                    f_out.write('\n')
                else:
                    translated_tokens = model.generate(
                        **tokenizer(line, return_tensors="pt").to(model.device),
                        forced_bos_token_id=tokenizer.lang_code_to_id[cfg.output.target_language_code],
                        bad_words_ids=None if len(bad_word_tokens) == 0 else bad_word_tokens,   # Needs to be None instead of empty, otherwise HF throws ValueError
                        max_length=cfg.model.max_length,
                        num_beams=getattr(cfg.generation, "num_beams", 10),
                        **dict(getattr(cfg.generation, "generator_kwargs", {}))
                    )[0]
                    f_out.write(tokenizer.decode(translated_tokens, skip_special_tokens=True) + '\n')
                f_out.flush()


if __name__ == "__main__":
    main()      # pylint: disable=no-value-for-parameter
