# TODO list

This file is a todo list of things with various levels of importance. List is in no particular order.




# Expansion Stuff

## Intra/InterConstruct Transport Cost Tax

Impose specific costs to all unstabilized inputs and outputs for a ComplexConstruct. This would effectively be like declaring "Everything in this construct is moved around via ___", and allow different levels of constructs to have different types of transport costs.

## Surface Restrictions

Classify Constructs by surface, prohibiting specific constructs (some resources, throwing worthless items into the void) from being used depending on the factory surface. Should allow for inter-surface factories.

## Expansion Beacon setups

What updates have to be done to handle diminishing beacons?


# Speedups

## Different solvers?

## Calculation orders

Should be: make an array and fill it rather than concat?
np.einsum vs np.kron
Transposing stuff outside function
Combining beacon operations together?


# Misc improvements

## Check and report missing non-optional tag in lua-api: https://lua-api.factorio.com/latest/prototypes/CharacterPrototype.html#mining_speed

## Decorator improvements for: Overloading .add(), type checking interface inputs

## Add my try catches for better debugging

Could do this with a decorator. Nope. this destroys time checking in vscode because Microsoft, a multi TRILLION dollar company, relies on one guy's open-source project, so its an "upstream issue."



# Progression accuracy

## Quality in science factories!

How do we handle more optimal processing in science factories (which may include running items through quality for material factories)?

WE CAN COMBINE A SCIENCE AT MATERIAL FACTORY INTO ONE. The material factory won't build the science factory but that allows for both to work.

## Better catalyst managment

Find a way of handling containers and catalysts properly. Also improve catalyst classification if possible (should iron really be a catalyst?... if not, why not?)
Idea: Catalysts are only catalysts when doing their catalytic process, for example: iron is only catalytic when being used in nuclear fuel. Best method to do this would be to split the catalyst cost off individual constructs, instead detect all catalytic loops and price each for the minimal amount of that each that exists. Catalysts only matter for priced indicies (which effect factory chain addition) and pricing in standard model (not space based models).
Only issue with this is deciding what construct NEEDS the catalytic element made to be allowed (otherwise unpriced costs could end up in our model) (idea: give each item an number, all energy sources have value of 0, otherwise an item has a value equal to 1+ the minimum value of the maximums of making it. The process with the maximum value is the 'final' process)
Doesn't work, breaks linearity


# Requires expansion api

## Productivity researches for recipe

https://www.factorio.com/blog/post/fff-376

## Expansion Decay constructs

Idea for decay: constructs with no cost that decays a decaying entity into its output.

## Fourth effect: Quality

"So, recycling an iron plate will just return an iron plate, with 25% chance"
Implement quality modules.