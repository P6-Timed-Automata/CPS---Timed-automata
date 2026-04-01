import xml.etree.ElementTree as ET

def create_simple_uppaal():
    # Root
    nta = ET.Element("nta")

    # Global declarations
    decl = ET.SubElement(nta, "declaration")
    decl.text = "clock x;\nint temp;"

    # Template
    template = ET.SubElement(nta, "template")

    name = ET.SubElement(template, "name")
    name.text = "SimpleTA"

    # --- Location 1 ---
    loc1 = ET.SubElement(template, "location", {
        "id": "id0",
        "x": "0",
        "y": "0"
    })
    ET.SubElement(loc1, "name").text = "A"

    inv1 = ET.SubElement(loc1, "label", {"kind": "invariant"})
    inv1.text = "x <= 10"

    # --- Location 2 ---
    loc2 = ET.SubElement(template, "location", {
        "id": "id1",
        "x": "200",
        "y": "0"
    })
    ET.SubElement(loc2, "name").text = "B"

    inv2 = ET.SubElement(loc2, "label", {"kind": "invariant"})
    inv2.text = "x <= 20"

    # Initial location
    init = ET.SubElement(template, "init", {"ref": "id0"})

    # --- Transition ---
    trans = ET.SubElement(template, "transition")

    ET.SubElement(trans, "source", {"ref": "id0"})
    ET.SubElement(trans, "target", {"ref": "id1"})

    # Guard
    guard = ET.SubElement(trans, "label", {"kind": "guard"})
    guard.text = "x >= 5"

    # Assignment (reset clock + update variable)
    assign = ET.SubElement(trans, "label", {"kind": "assignment"})
    assign.text = "x = 0, temp = 15"

    # System declaration
    system = ET.SubElement(nta, "system")
    system.text = "Process = SimpleTA();system Process;"

    return ET.ElementTree(nta)


# Run and save
tree = create_simple_uppaal()
tree.write("simple_model.xml", encoding="utf-8", xml_declaration=True)



# DATA structures

# Node : states, invariant

# Edge: source, destination, guard

# Positions, coordinate points, with node id

#FUNCTIONS

#Symbol assignment: for each symbol, assign a temperature. The temperature is the median of the bin interval

#Guard:  extract  the guard from TAG guard (a[10,20]  --> t>= 10 && t<=20

#Invariant: find the maximum upperbound guard value from the outgoing edges

#Coordinates: extract the coordinates from graphiz




