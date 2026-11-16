# Import mjlab before any robot module starts executing. `import mjlab` runs its
# task-plugin loader, which imports mjlab_microduck.tasks, which imports the robot
# constants. If a robot constants module were the FIRST thing imported (a test, a
# script), that chain would re-enter it half-initialized and the plugin would fail
# to register. Doing it here makes the robot modules safe to import first.
import mjlab  # noqa: F401
