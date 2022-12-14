import os
import sys
import copy
import logging
import glob

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from cai_garland.utils.translator import Translator


def interactive(translator, _mode_cfg, _generation_cfg, target_language_code):
    print("Interactive Tibetan translation...")
    while True:
        print("===")
        bo_text = input("Tibetan (or type exit): ")
        if bo_text == "exit":
            break
        print(translator.translate(bo_text, target_language_code=target_language_code))


def batch(translator, mode_cfg, generation_cfg, target_language_code):
    if mode_cfg.input_glob is None:
        raise ValueError("Specify an input file (or glob) in the mode.input_glob setting. You can do this from the "
                         "command line.")
    os.makedirs(mode_cfg.output_dir, exist_ok=True)
    in_fns = glob.glob(mode_cfg.input_glob)
    original_cfg = copy.deepcopy(generation_cfg)
    for in_fn in (files_pbar := tqdm(in_fns)):
        files_pbar.set_description(os.path.basename(in_fn))
        with open(in_fn, encoding=mode_cfg.input_encoding) as in_f:
            bo_text = in_f.read()

        generation_cfg = copy.deepcopy(original_cfg)
        in_cfg_fn = os.path.join(os.path.dirname(in_fn), os.path.splitext(os.path.basename(in_fn))[0] + '.config.yaml')
        if os.path.isfile(in_cfg_fn):
            in_cfg = OmegaConf.load(in_cfg_fn)
            generation_cfg = OmegaConf.merge(generation_cfg, in_cfg)

        out_fn = os.path.join(
            mode_cfg.output_dir, os.path.splitext(os.path.basename(in_fn))[0] + '.' + mode_cfg.output_extension)
        translation_kwargs = {}
        translation_kwargs["retrospective_decoding"] = generation_cfg.generation.get("retrospective_decoding", False)
        translation_kwargs["retrospective_registers"] = \
            translator.model.encoder.config.model_type == "siamese-encoder" \
            and generation_cfg.generation.get("use_registers_for_retrospective_decoding", True)
        translation_kwargs["retrospection_window"] = generation_cfg.generation.get("retrospection_window", None)
        translation_kwargs["contextual_decoding"] = hasattr(generation_cfg.generation, "pooled_context")
        if translation_kwargs["contextual_decoding"]:
            translation_kwargs["context_window_words"] = generation_cfg.generation.pooled_context.context_window.words
            translation_kwargs["context_window_characters"] = \
                generation_cfg.generation.pooled_context.context_window.characters

        with open(out_fn, mode='w') as out_f:
            translator.hard_segmenter = instantiate(generation_cfg.segmentation.hard_segmentation)
            translator.preprocessors = [
                instantiate(preproc_func)
                for preproc_func in generation_cfg.processing.preprocessing
            ]
            translator.soft_segmenter = instantiate(
                generation_cfg.segmentation.soft_segmentation, translator=translator)
            translator.soft_segment_combiner_config = getattr(
                generation_cfg.segmentation, "soft_segment_combiner", None)
            translator.soft_segment_preprocessors = [
                instantiate(preproc_func)
                for preproc_func in generation_cfg.processing.get("soft_segment_preprocessing", [])
            ]
            translator.postprocessors = [
                instantiate(preproc_func)
                for preproc_func in generation_cfg.processing.postprocessing
            ]

            for src_segment, tgt_segment in translator.batch_translate(
                bo_text,
                tqdm=tqdm,
                hard_segmenter_kwargs=dict(generation_cfg.segmentation.hard_segmenter_kwargs),
                soft_segmenter_kwargs=dict(generation_cfg.segmentation.soft_segmenter_kwargs),
                throw_translation_errors=not mode_cfg.skip_long_inputs,
                target_language_code=target_language_code,
                generator_kwargs=dict(generation_cfg.generation.get("generator_kwargs", {})),
                **translation_kwargs
            ):
                if mode_cfg.output_parallel_translation:
                    out_f.write(src_segment + '\n')
                out_f.write('\n')
                out_f.write(tgt_segment + '\n')
                out_f.write('\n')
                out_f.flush()


@hydra.main(version_base="1.2", config_path="translation_config", config_name="translate")
def main(cfg):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    translator = Translator(os.path.join(cfg.model.model_ckpt, cfg.model.model_size))
    translator.num_beams = cfg.generation.generation.num_beams
    if hasattr(cfg.model, "decoding_length"):
        translator.decoding_length = cfg.model.decoding_length
    if hasattr(cfg.generation.generation, "pooled_context"):
        translator.prepare_context_encoder(cfg.generation.generation.pooled_context.context_encoder.hf_model_name)
    target_language_code = getattr(cfg, "target_language_code", None)

    if cfg.cuda:
        translator.cuda()

    if target_language_code is not None and hasattr(cfg, "word_exclusion"):
        translator.bad_words = getattr(cfg.word_exclusion, target_language_code, [])

    instantiate(cfg.mode.process_func, translator, cfg.mode, cfg.generation, target_language_code)


if __name__ == "__main__":
    main()      # pylint: disable=no-value-for-parameter
