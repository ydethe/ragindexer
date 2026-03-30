mkdir -p htmldoc/cisaggregator
python -m pdoc --html --force --config latex_math=True -o htmldoc cisaggregator
python -m coverage html -d htmldoc/coverage --rcfile tests/coverage.conf
python -m coverage xml -o htmldoc/coverage/coverage.xml --rcfile tests/coverage.conf
python -m docstr-coverage src/cisaggregator -miP -sp -is -idel --skip-file-doc --badge=htmldoc/cisaggregator/doc_badge.svg
python -m genbadge coverage -l -i htmldoc/coverage/coverage.xml -o htmldoc/cisaggregator/cov_badge.svg
python -m http.server -d htmldoc 12345
