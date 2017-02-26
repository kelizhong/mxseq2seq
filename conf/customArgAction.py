import argparse
class AppendTupleWIthoutDefault(argparse._AppendAction):
    def __call__(self, parser, namespace, values, option_string=None):

        items = argparse._copy.copy(argparse._ensure_value(namespace, self.dest, []))
        try:
            self._not_first
        except AttributeError:
            self._not_first = True
            del items[:]
        items.append(tuple(values))
        setattr(namespace, self.dest, items)