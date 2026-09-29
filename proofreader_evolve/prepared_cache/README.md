# Historical prepared-brain caches

These files belong to the removed graph-edit workflow. Current fixed-pool
scorer evolution neither reads nor rebuilds them; it uses feature_tables/
prepared from labeled _add.pkl caches.

Files were retained to avoid deleting historical data. Their pickled classes
require the original code version to load. They are not interchangeable with
native candidate tables, and removing them will not trigger an automatic rebuild
in the current workflow.
