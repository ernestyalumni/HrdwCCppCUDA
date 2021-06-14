[Dmitry Soshnikov, "Writing a Memory Allocator"](http://dmitrysoshnikov.com/compilers/writing-a-memory-allocator/)

## Mutator, Allocator, Collector

**Mutator** is our user-program, where we create objects for own purposes.
- All other modules should *respect* the Mutator's view on the object graph.
- e.g. under no circumstances a **Collector** can recalim an *alive* object

**Allocator** - mutator doesn't allocate objects by itself; instead it delegates this general task to the Allocator module.

