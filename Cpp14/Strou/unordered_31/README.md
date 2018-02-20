# `unordered_map`, `unordered_set`  

`<unordered_map>`  

cf. pp. 886, Ch. 31 *STL Containers*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.   

cf. pp. 886, Sec. 31.2 "Container Overview", Ch. 31 *STL Containers*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.   

Container holds sequence of objects.

Containers can be categorized into:  
* *Sequence containers* provide access to (half-open) sequences of elements.  
* *Associative containers* provide associative lookup based on a key.  

In addition, standard library (STL) provides types of objects that hold elements while not offering all facilities of sequence or associative containers:  

### Unordered Associative Containers (Sec.iso.23.5.2)  

`H` is hash function type, `E` equality test, `A` allocator type
- `unordered_map<K,V,H,E,A>` - An unordered map from `K` to `V`   
- `unordered_multimap<K,V,H,E,A>` - An unordered map from `K` to `V`; duplicate keys allowed    
- `unordered_set<K,H,E,A>` - An unordered set of `K`    
- `unordered_multiset<K,H,E,A>` - An unordered set of `K`; duplicate keys allowed  

These containers are implemented as hash tables with linked overflow.  (cf. [Hashing chain, uni freiburg](http://gki.informatik.uni-freiburg.de/teaching/ss11/theoryI/07_Hashing_Chaining.pdf)  

Default hash function type, `H`, for type `K` is `std::hash<K>` (Sec.31.4.3.2)  
Default equality function type, `E`, for type `K` is `std::equal_to<K>` (Sec.33.4); used to decide whether 2 objects with same hash code are equal.  

Associative containers are linked structures (trees) with nodes of their `value_type` (in notation above, `pair<const K,V>` for maps and `K` for sets).  
Unordered container need not have ordering relation for its elements (e.g. `<`) and uses hash function instead (Sec.31.2.2.1).  Sequence of an unordered container doesn't guarantee order.  
`multimap` differs from `map` in that key value may occur many times.  

cf. pp. 893, Sec. 31.3 "Operations Overview", Ch. 31 *STL Containers*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.   
   

Associative container: `key_type, mapped_type, ?[], ?at(), lower_bound(), upper_bound(), equal_range(), find(), count(), emplace_hint()`  
<- Hashed container: `key_equal(), hasher, hash_function(), key_equal()` bucket interface  
<- `unordered_map, unordered_set, unordered_multimap, unordered_multiset`  

* `unordered_*` associative container doesn't provide `<,<=,>`, or `>=`  

STL operations complexity gurantees  

| **Standard Container Operation Complexity** | 	 | 		| 		| 		|		|
| :------------------------------------------ | :--- | :--- | :---- | :---- | :---- |
| `unordered_map` 							| const+ | const+ |     | 		| For |  
| `unordered_multimap` 						 |  | const+ |     | 		| For |
| `unordered_set` 							|  | const+ |     | 		| For |  
| `unordered_multiset` 						 |  | const+ |     | 		| For |

"For" means "forward iterator".  
const means operation takes amount of time that doesn't depend on number of elements in the container, constant time or *O(1)*,  

 
cf. pp. 913, Sec. 31.4.3.2 "Unordered Associative Containers", Ch. 31 *STL Containers*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.   

Unordered associative containers (`unordered_map, unordered_set, unordered_multimap, unordered_multiset`) are hash tables.  
