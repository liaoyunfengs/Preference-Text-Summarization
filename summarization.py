import sys
import time

import tensorflow as tf
import batch_reader
import data
import seq2seq_attention_decode
import seq2seq_attention_model
#import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

mode_in = 'train'  #选择训练集或者测试集
article_key = 'article'
abstract_key = 'abstract'  # 标题生成：headline 摘要生成：abstract
data_path = 'binary_yanbao_output_300'
#data_path = 'binary_data'  # 原始data_new
vocab_path = 'yanbao_vocab_all'
log_root = 'log_root'
train_dir = 'train'
eval_dir = 'eval'
decode_dir = 'decode'
beam_size = 8  #beam search top k for decode

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string('data_path',
                           data_path, 'Path expression to tf.Example.')
tf.flags.DEFINE_string('vocab_path',
                           vocab_path, 'Path expression to text vocabulary file.')
tf.flags.DEFINE_string('article_key', article_key,
                           'tf.Example feature key for article.')
tf.flags.DEFINE_string('abstract_key', abstract_key,
                           'tf.Example feature key for abstract.')
tf.flags.DEFINE_string('log_root', log_root, 'Directory for model root.')
tf.flags.DEFINE_string('train_dir', train_dir, 'Directory for train.')
tf.flags.DEFINE_string('eval_dir', eval_dir, 'Directory for eval.')
tf.flags.DEFINE_string('decode_dir', decode_dir, 'Directory for decode summaries.')
tf.flags.DEFINE_string('mode', mode_in, 'train/eval/decode mode')
tf.flags.DEFINE_integer('max_run_steps', 10000000,
                            'Maximum number of run steps.')#10000000
tf.flags.DEFINE_integer('max_article_sentences', 100,
                            'Max number of first sentences to use from the '
                            'article') #原来为2 训练24800次的时候好像是10
tf.flags.DEFINE_integer('max_abstract_sentences', 100,
                            'Max number of first sentences to use from the '
                            'abstract')  #原来为100
tf.flags.DEFINE_integer('beam_size', beam_size,
                            'beam size for beam search decoding.')
tf.flags.DEFINE_integer('eval_interval_secs', 60, 'How often to run eval.')
tf.flags.DEFINE_integer('checkpoint_secs', 60, 'How often to checkpoint.')
tf.flags.DEFINE_bool('use_bucketing', True,
                         'Whether bucket articles of similar length.')  #原来是False
tf.flags.DEFINE_bool('truncate_input', True,
                         'Truncate inputs that are too long. If False, ' #原来是False
                         'examples that are too long are discarded.')
tf.flags.DEFINE_integer('num_gpus', 0, 'Number of gpus used.')#有显卡 1  无显卡 0
tf.flags.DEFINE_integer('random_seed', 111, 'A seed value for randomness.')


def _RunningAvgLoss(loss, running_avg_loss, summary_writer, step, decay=0.999):
    """Calculate the running average of losses."""
    if running_avg_loss == 0:
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)
    loss_sum = tf.Summary()
    loss_sum.value.add(tag='running_avg_loss', simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)
    sys.stdout.write('running_avg_loss: %f\n' % running_avg_loss)
    return running_avg_loss


def _Train(model, data_batcher):
    """Runs model training."""
    with tf.device('/cpu:0'):
        model.build_graph()
        print('build_graph over')
        saver = tf.train.Saver()
        print('saver over')
        # Train dir is different from log_root to avoid summary directory
        # conflict with Supervisor.
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
        print('summary over')
        sv = tf.train.Supervisor(logdir=FLAGS.log_root,
                                 is_chief=True,
                                 saver=saver,
                                 summary_op=None,
                                 save_summaries_secs=60,
                                 save_model_secs=FLAGS.checkpoint_secs,
                                 global_step=model.global_step)
        sess = sv.prepare_or_wait_for_session(config=tf.ConfigProto(
             allow_soft_placement=True))  #原来是True
        print('session over')
        #sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        running_avg_loss = 0
        step = 0
        print('start train')
        while not sv.should_stop() and step < FLAGS.max_run_steps:
            (article_batch, abstract_batch, targets, article_lens, abstract_lens,
             loss_weights, _, _) = data_batcher.NextBatch()
            (_, summaries, loss, train_step) = model.run_train_step(
                sess, article_batch, abstract_batch, targets, article_lens,
                abstract_lens, loss_weights)

            summary_writer.add_summary(summaries, train_step)
            running_avg_loss = _RunningAvgLoss(
                running_avg_loss, loss, summary_writer, train_step)
            step += 1
            print('train step=', step)
            if step % 100 == 0:
                summary_writer.flush()
        sv.Stop()
        return running_avg_loss


def _Eval(model, data_batcher, vocab=None):
    """Runs model eval."""
    model.build_graph()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    running_avg_loss = 0
    step = 0
    while True:
        time.sleep(FLAGS.eval_interval_secs)
        try:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
            continue

        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to eval yet at %s', FLAGS.train_dir)
            continue

        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        (article_batch, abstract_batch, targets, article_lens, abstract_lens,
         loss_weights, _, _) = data_batcher.NextBatch()
        (summaries, loss, train_step) = model.run_eval_step(
            sess, article_batch, abstract_batch, targets, article_lens,
            abstract_lens, loss_weights)
        tf.logging.info(
            'article:  %s',
            ' '.join(data.Ids2Words(article_batch[0][:].tolist(), vocab)))
        tf.logging.info(
            'abstract: %s',
            ' '.join(data.Ids2Words(abstract_batch[0][:].tolist(), vocab)))

        summary_writer.add_summary(summaries, train_step)
        running_avg_loss = _RunningAvgLoss(
            running_avg_loss, loss, summary_writer, train_step)
        if step % 100 == 0:
            summary_writer.flush()


def main(unused_argv):
    vocab = data.Vocab(FLAGS.vocab_path, 1000000)
    #Check for presence of required special tokens.
    assert vocab.CheckVocab(data.PAD_TOKEN) > 0
    assert vocab.CheckVocab(data.UNKNOWN_TOKEN) >= 0
    assert vocab.CheckVocab(data.SENTENCE_START) > 0
    assert vocab.CheckVocab(data.SENTENCE_END) > 0

    batch_size = 4
    if FLAGS.mode == 'decode':
        batch_size = FLAGS.beam_size

    hps = seq2seq_attention_model.HParams(
        mode=FLAGS.mode,  # train, eval, decode
        min_lr=0.01,  # min learning rate.
        lr=0.15,  # learning rate
        batch_size=batch_size,
        enc_layers=4,
        enc_timesteps=500,   #120  #400是32G内存极限
        dec_timesteps=70,   #30
        min_input_len=2,  # discard articles/summaries < than this
        num_hidden=256,  # for rnn cell 256
        emb_dim=128,  # If 0, don't use embedding 128
        max_grad_norm=2,
        num_softmax_samples=4096)  # If 0, no sampled softmax.  4096

    batcher = batch_reader.Batcher(FLAGS.data_path, vocab, hps, FLAGS.article_key,FLAGS.abstract_key, FLAGS.max_article_sentences,FLAGS.max_abstract_sentences, bucketing=FLAGS.use_bucketing,truncate_input=FLAGS.truncate_input)
    tf.set_random_seed(FLAGS.random_seed)
    print('batch_read over')
    if hps.mode == 'train':
        model = seq2seq_attention_model.Seq2SeqAttentionModel(
            hps, vocab, num_gpus=FLAGS.num_gpus)
        print('model over')
        _Train(model, batcher)
    elif hps.mode == 'eval':
        model = seq2seq_attention_model.Seq2SeqAttentionModel(
            hps, vocab, num_gpus=FLAGS.num_gpus)
        _Eval(model, batcher, vocab=vocab)
    elif hps.mode == 'decode':
        decode_mdl_hps = hps
        # Only need to restore the 1st step and reuse it since
        # we keep and feed in state for each step's output.
        decode_mdl_hps = hps._replace(dec_timesteps=1)
        model = seq2seq_attention_model.Seq2SeqAttentionModel(
            decode_mdl_hps, vocab, num_gpus=FLAGS.num_gpus)
        decoder = seq2seq_attention_decode.BSDecoder(model, batcher, hps, vocab)
        decoder.DecodeLoop()


if __name__ == '__main__':
    tf.app.run()
