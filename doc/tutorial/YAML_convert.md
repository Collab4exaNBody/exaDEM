# How to convert to your own enum type from input file

## Example 

Given your enum type named `my_type`, you have to define a specialization of the structure convert and define `encode` and `decode`. The following lines are an example of this structure, the input parameter is read as a string, and we use another function to convert it to my type. You can directly convert it into double or int by defining it in the function `as`.


```
namespace YAML
{
  using exaDEM::my_type;
  template<> struct convert<my_type>
  {
    static inline Node encode(my_type& in)
    {
      Node node;
      std::string str = exaDEM::convert_to_string(in);
      node.push_back(str);
      return node;
    }
    static inline bool decode(const Node& node, my_type& out)
    {
      out = exaDEM::convert_to_my_type(node.as<std::string>());
      return true;
    }
  };
}
```

## Example for a vector of your datatype

This is the version to read and write a vector of an enum type. It works similarly to the previous. Note that the previous has to be defined in this case.

```
namespace YAML
{
  using exaDEM::my_type;
  template<> struct convert<std::vector<my_type>>
  {
    static inline Node encode(std::vector<my_type>& in)
    {
      Node node;
			constexpr int size = 6;
      for(int i = 0; i < size ; i++)
      {
        std::string str = exaDEM::convert_to_string(in[i]);
        node.push_back(str);
      }
      return node;
    }
    static inline bool decode(const Node& node, std::vector<my_type>& out)
    {
			constexpr int size = 6;
      out.resize(size);
      for(int i = 0; i < size ; i++) 
			{
        out[i] = node[i].as<my_type>();
			}
      return true;
    }
  };
}
```
