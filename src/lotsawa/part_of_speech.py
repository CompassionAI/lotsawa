import os
import sys
import glob
import hydra
import pickle
import logging

from tqdm.auto import tqdm
from hydra.utils import instantiate
from cai_manas.part_of_speech.pos_tagger import PartOfSpeechTagger


def interactive(tagger, _mode_cfg):
    print("Interactive Tibetan part-of-speech tagging...")
    while True:
        print("===")
        bo_text = input("Tibetan (or type exit): ")
        if bo_text == "exit":
            break
        res = tagger.tag(bo_text)
        for word, tag in zip(res["words"], res["tags"]):
            print(word, tag)


def batch(tagger, mode_cfg):
    if mode_cfg.input_glob is None:
        raise ValueError("Specify an input file (or glob) in the mode.input_glob setting. You can do this from the "
                         "command line.")
    to_pickle = mode_cfg.output_pickle_file
    os.makedirs(mode_cfg.output_dir, exist_ok=True)
    in_fns = glob.glob(mode_cfg.input_glob)
    for in_fn in (files_pbar := tqdm(in_fns)):
        files_pbar.set_description(os.path.basename(in_fn))
        with open(in_fn, encoding=mode_cfg.input_encoding) as in_f:
            bo_text = in_f.read()

        out_fn = os.path.join(
            mode_cfg.output_dir, os.path.splitext(os.path.basename(in_fn))[0] + '.' + mode_cfg.output_extension)

        if to_pickle:
            for_pickle = []
        with open(out_fn, mode='w') as out_f:
            tagger.segmenter = instantiate(mode_cfg.segmenter.segmenter)
            tagger.preprocessors = [instantiate(preproc_func) for preproc_func in mode_cfg.get("preprocessing", [])]

            for segments, tags in tagger.batch_tag(
                bo_text,
                tqdm=tqdm,
                segmenter_kwargs=dict(mode_cfg.segmenter.kwargs),
                throw_encoder_errors=not mode_cfg.skip_long_inputs
            ):
                out_f.write(' '.join([f"{segment}[{tag}]" for segment, tag in zip(segments, tags)]))
                out_f.write('\n')
                out_f.write('\n')
                out_f.flush()
                if to_pickle:
                    for_pickle.append((segments, tags))
        if to_pickle:
            logging.info("Dumping pickle file")
            with open(out_fn + '.pkl', 'wb') as out_f:
                pickle.dump(for_pickle, out_f)


@hydra.main(version_base="1.2", config_path="token_classification_config", config_name="part_of_speech")
def main(cfg):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    tagger = PartOfSpeechTagger(cfg.model.model_ckpt)

    instantiate(cfg.mode.process_func, tagger, cfg.mode)


if __name__ == "__main__":
    main()      # pylint: disable=no-value-for-parameter
