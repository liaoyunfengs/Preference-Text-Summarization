from six.moves import xrange
import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_bool('normalize_by_length', True, 'Whether to normalize')


class Hypothesis(object):
    """Defines a hypothesis 假设 during beam search."""

    def __init__(self, tokens, log_prob, state):
        """Hypothesis constructor.假设构造函数
    Args:
      tokens: start tokens for decoding.
      log_prob: log prob of the start tokens 命令, usually 1.
      state: decoder initial states.
      戳标记：开始戳标记为decoding。
       问题：问题_ log日志”的戳标记开始，usually 1。
       状态：初始decoder QE
    """
        self.tokens = tokens
        self.log_prob = log_prob
        self.state = state

    def Extend(self, token, log_prob, new_state):
        """Extend the hypothesis with result from latest step.
    Args:
      token: latest token from decoding.
      log_prob: log prob of the latest decoded tokens.
      new_state: decoder output state. Fed to the decoder for next step.
    Returns:
      New Hypothesis with the results from latest step.
    """
        return Hypothesis(self.tokens + [token], self.log_prob + log_prob,
                          new_state)

    @property
    def latest_token(self):
        return self.tokens[-1]

    def __str__(self):
        return ('Hypothesis(log prob = %.4f, tokens = %s)' % (self.log_prob,
                                                              self.tokens))


class BeamSearch(object):
    """Beam search."""

    def __init__(self, model, beam_size, start_token, end_token, max_steps):
        """Creates BeamSearch object.
    Args:
      model: Seq2SeqAttentionModel.
      beam_size: int.

      start_token: int, id of the token to start decoding with
      令牌开始解码的ID
      end_token: int, id of the token that completes an hypothesis
      max_steps: int, upper limit on the size of the hypothesis
    """
        self._model = model
        self._beam_size = beam_size
        self._start_token = start_token
        self._end_token = end_token
        self._max_steps = max_steps

    def BeamSearch(self, sess, enc_inputs, enc_seqlen):
        """Performs beam search for decoding.
    Args:
      sess: tf.Session, session
      enc_inputs: ndarray of shape (enc_length, 1), the document ids to encode
      enc_seqlen: ndarray of shape (1), the length of the sequnce
    Returns:
      hyps: list of Hypothesis, the best hypotheses found by beam search,
          ordered by score 假设列表，波束搜索发现的最佳假设，按分数排序
    """

        # Run the encoder and extract the outputs and final state.
        enc_top_states, dec_in_state = self._model.encode_top_state(
            sess, enc_inputs, enc_seqlen)
        # Replicate the initial states K times for the first step.
        hyps = [Hypothesis([self._start_token], 0.0, dec_in_state)
                ] * self._beam_size
        results = []

        steps = 0
        while steps < self._max_steps and len(results) < self._beam_size:
            latest_tokens = [h.latest_token for h in hyps]
            states = [h.state for h in hyps]

            topk_ids, topk_log_probs, new_states = self._model.decode_topk(
                sess, latest_tokens, enc_top_states, states)
            # Extend each hypothesis.
            all_hyps = []
            # The first step takes the best K results from first hyps. Following
            #第一步从第一个假设得到最好的K结果。跟随
            # steps take the best K results from K*K hyps.
            num_beam_source = 1 if steps == 0 else len(hyps)
            for i in xrange(num_beam_source):
                h, ns = hyps[i], new_states[i]
                for j in xrange(self._beam_size * 2):
                    all_hyps.append(h.Extend(topk_ids[i, j], topk_log_probs[i, j], ns))

            # Filter and collect any hypotheses that have the end token.
            #过滤并收集具有终结标记的任何假设
            hyps = []
            for h in self._BestHyps(all_hyps):
                if h.latest_token == self._end_token:
                    # Pull the hypothesis off the beam if the end token is reached.
                    results.append(h)
                else:
                    # Otherwise continue to the extend the hypothesis.
                    hyps.append(h)
                if len(hyps) == self._beam_size or len(results) == self._beam_size:
                    break

            steps += 1

        if steps == self._max_steps:
            results.extend(hyps)

        return self._BestHyps(results)

    def _BestHyps(self, hyps):
        """Sort the hyps based on log probs and length.
    Args:
      hyps: A list of hypothesis.
    Returns:
      hyps: A list of sorted hypothesis in reverse log_prob order.
    """
        # This length normalization is only effective for the final results.
        if FLAGS.normalize_by_length:
            return sorted(hyps, key=lambda h: h.log_prob / len(h.tokens), reverse=True)
        else:
            return sorted(hyps, key=lambda h: h.log_prob, reverse=True)
