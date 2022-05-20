import npnd
import npnd.test_util as ntu

class TestCase(ntu.TestCase):
  def test_basic(self):
    ntu.check_eq(1, 1)

if __name__ == '__main__':
  ntu.main()
