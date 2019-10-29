from absl import app


def main(_):
    from config import FLAGS
    if FLAGS.competition == 'chip':
        from chip.preproc import preproc_entry
        from chip.train import train_entry
        from chip.test import test_entry
        from chip.test import eval_entry

        if FLAGS.mode == 'data':
            preproc_entry()
        elif FLAGS.mode == 'train':
            train_entry()
        elif FLAGS.mode == 'test':
            test_entry()
        elif FLAGS.mode == 'eval':
            eval_entry()
        else:
            raise AttributeError("unknown mode")


if __name__ == '__main__':
    app.run(main)
