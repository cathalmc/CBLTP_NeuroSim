import dash
from dash import Dash, html, dcc, callback, Output, Input, State, no_update
import dash_cytoscape as cyto
import networkx as nx
import numpy as np
import json 
import ast
import os
from dash import MATCH, ALL


default_stylesheet = [
    # Base node style
    {"selector": "node", "style": {
        "label": "data(label)",
        "width": 30, "height": 30,
        "color": "black",
        "background-color": "#d0d0d0",
        "font-size": "12px"
    }},
    # Base edge style
    {"selector": "edge", "style": {
        "curve-style": "bezier",
        "target-arrow-shape": "triangle",
        "label": "data(label)",
        "font-size": "10px",
        "text-background-color": "white",
        "text-background-opacity": 1,
        "color": "black"
    }},
    # Voltage only (red): must go AFTER base style but BEFORE the more general rules
    {"selector": '[voltageDependent = "True"][ligandDependent = "False"]', "style": {
        "line-color": "red",
        "target-arrow-color": "red"
    }},
    # Ligand only (green)
    {"selector": '[voltageDependent = "False"][ligandDependent = "True"]', "style": {
        "line-color": "green",
        "target-arrow-color": "green"
    }},
    # Both voltage and ligand dependent (yellow) — most specific, goes last
    {"selector": '[voltageDependent = "True"][ligandDependent = "True"]', "style": {
        "line-color": "yellow",
        "target-arrow-color": "yellow"
    }},
]


class GraphStore:
    #going to make this a generic class, then ion channels/ gap junctions can be sub classes
    def __init__(self,baseFolder=None):
        self.G = nx.DiGraph()
        self.currentId = 0
        self.metaData = {}
        if baseFolder is None:
            baseFolder = "nets"
        self.baseFolder = baseFolder

    def add_node(self, label, pos=None):
        self.G.add_node(self.currentId, label=label, position=pos)
        self.currentId += 1
    
    def add_edge(self, source, target, label=None, synType=None, **kwargs):
        if label is None:
            label = f"{source}-{target}"

        # Set default attributes
        edgeAttrs = {
            "label": label,
            "synapseType":synType,
            "conductance": 1.0,
        }

        # Update with any passed kwargs
        edgeAttrs.update(kwargs)

        # Add the edge with all attributes
        self.G.add_edge(source, target, **edgeAttrs)

    def get_elements(self):
        nodes = []
        for n in self.G.nodes:
            data = {"id": str(n), "label": str(self.G.nodes[n].get("label", n))}
            pos = self.G.nodes[n].get("position")
            if type(pos) == str:
                pos = ast.literal_eval(pos)
            node = {"data": data}
            if pos:
                node["position"] = {"x": pos[0], "y": pos[1]}
            nodes.append(node)
        edges = []
        for u, v, d in self.G.edges(data=True):
            edge = {"data": {"id": f"{u}_{v}", "label":str(d["label"]), "source": str(u), "target": str(v),
                      "synapseType":str(d["synapseType"]),"conductance":d["conductance"]
                      }}
            edges.append(edge)
            
        return nodes + edges

    def update_node_positions(self, posDict):
        for nodeId, pos in (posDict or {}).items():
            try:
                nodeKey = int(nodeId) if isinstance(nodeId, str) and nodeId.isdigit() else nodeId
            except Exception:
                nodeKey = nodeId
            if nodeKey in self.G.nodes:
                self.G.nodes[nodeKey]["position"] = (pos["x"], pos["y"])

    def convertToDict(self):
        return {}

    def loadFromJson(self,filename):
        if not filename.endswith(".json"):
            filename = filename + ".json"
        filename = os.path.join(self.baseFolder,filename)
        with open(filename, "r") as f:
            data = json.load(f)

        self.metaData = data["metaData"]

        n0 = self.currentId
        newG = nx.readwrite.json_graph.node_link_graph(data["graphData"],edges="links")
        for node,nodeData in newG.nodes.items():
            self.add_node(nodeData["label"],nodeData["position"],
                          )

        for (s,t),edgeData in newG.edges.items():
            s,t = s+n0, t+n0
            self.add_edge(s,t,**edgeData)
        return self
    
    def saveToJson(self,filename):
        filename = os.path.join(self.baseFolder,filename)
        if not filename.endswith(".json"):
            filename = filename + ".json"
        gData = self.convertToDict()
        with open(filename, "w") as f:
            json.dump(gData,f)


class neuralNet(GraphStore):
    def __init__(self,baseFolder=None):
        super().__init__(baseFolder)
        self.metaData = {
            "name":"test", #str
            "mechType":"neuralNet",
        }

    def convertToDict(self):
        data = {
            "metaData":self.metaData,
            "graphData":nx.node_link_data(self.G,edges = "links")
        }
        return data
    
    def loadFromJson(self, filename):
        super().loadFromJson(filename)
        assert self.metaData["mechType"] == "neuralNet", f"warning incorrect mechType, should be neuralNet, {self.mechType}"
        return self
    


class graphApp:
    def __init__(self,gStore,styleSheet):
        self.gStore = gStore
        self.appType = gStore.metaData["mechType"] 

        self.styleSheet = styleSheet
        self.app = Dash(__name__)
        self.__buildLayout()
        self.__registerCallbacks()

    def __registerCallbacks(self):
        @self.app.callback(
            Output("btn-remove-node", "disabled"),
            Output("btn-remove-edge", "disabled"),
            Input("selectedElementData", "data"),
            Input("selectedNodes", "data")
        )
        def toggle_remove_buttons(selElem, selNodes):
            nodeDisabled = True
            edgeDisabled = True
            if selElem and selElem.get("type") == "node":
                nodeDisabled = False
            elif selNodes:
                nodeDisabled = False
            if selElem and selElem.get("type") == "edge":
                edgeDisabled = False
            elif selNodes and len(selNodes) == 2:
                try:
                    src = int(selNodes[0]) if isinstance(selNodes[0], str) and selNodes[0].isdigit() else selNodes[0]
                    tgt = int(selNodes[1]) if isinstance(selNodes[1], str) and selNodes[1].isdigit() else selNodes[1]
                except Exception:
                    src, tgt = selNodes
                if self.gStore.G.has_edge(src, tgt):
                    edgeDisabled = False
            return nodeDisabled, edgeDisabled


        @self.app.callback(Output("nodePositions", "data"), Input("cytoCanvas", "elements"))
        def capturePositions(els):
            if not els:
                return no_update
            pos = {
                d["id"]: e["position"]
                for e in els
                for d in [e.get("data", {})]
                if e.get("position") and "id" in d
            }
            if not pos:
                return no_update
            self.gStore.update_node_positions(pos)
            return pos


        @self.app.callback(
            Output("selectedElementData", "data"),
            Output("selectedNodes", "data"),
            Input("cytoCanvas", "tapNode"),
            Input("cytoCanvas", "tapEdge"),
            State("selectedNodes", "data"),
            State("selectedElementData", "data"),
            prevent_initial_call=True
        )
        def update_selection(tapNode, tapEdge, selected, currentSelElem):
            sel = list(selected or [])
            triggered = dash.callback_context.triggered
            if not triggered:
                return no_update, no_update

            prop = triggered[0]["prop_id"]  # like "cytoCanvas.tapNode" or "cytoCanvas.tapEdge"

            def makeNodeSel(nodeData):
                nid = nodeData["id"] #how does node id relate to graph node id? 
                newSel = sel + [nid] 
                newSel = newSel[-2:]
                if len(newSel) == 2 and newSel[0] == newSel[1]:
                    newSel = [newSel[1]]
                return {"type": "node", "data": nodeData}, newSel

            def makeEdgeSel(edgeData):
                return {"type": "edge", "data": edgeData}, sel

            if prop.endswith("tapNode") and tapNode:
                node_id = tapNode["data"]["id"]
                if currentSelElem and currentSelElem.get("type") == "node" and currentSelElem["data"].get("id") == node_id:
                    # deselect the node
                    #sel = [s for s in sel if s != node_id]
                    sel = []
                    return None, sel
                
                return makeNodeSel(tapNode["data"])

            if prop.endswith("tapEdge") and tapEdge:
                edge_id = tapEdge["data"]["id"]
                if currentSelElem and currentSelElem.get("type") == "edge" and currentSelElem["data"].get("id") == edge_id:
                    return None, sel
                return makeEdgeSel(tapEdge["data"])

            return no_update, no_update



        @self.app.callback(
            Output("cytoCanvas", "elements"),
            Output("selectedNodes", "data",allow_duplicate=True),
            Output("selectedElementData", "data",allow_duplicate=True),
            Input("btn-add-node", "n_clicks_timestamp"),
            Input("btn-add-edge", "n_clicks_timestamp"),
            Input("btn-remove-node", "n_clicks_timestamp"),
            Input("btn-remove-edge", "n_clicks_timestamp"),
            Input("update-element-props", "n_clicks_timestamp"),
            State("cytoCanvas", "elements"),
            State("selectedNodes", "data"),
            prevent_initial_call=True
        )
        def update_elements(btnAdd, btnAddEdge,btnRemoveNode,btnRemoveEdge,btnUpdateData, elements, selectedNodes): #,
            if elements is None: return no_update, no_update, no_update

            btns = [btnAdd, btnAddEdge, btnRemoveNode, btnRemoveEdge, btnUpdateData]
            btns = [int(b or 0) for b in btns]  # None→0, keep full precision

            if max(btns) == 0:
                return no_update, no_update, no_update  # nothing clicked since app start

            possibleScenarios = {0:"addNode",1:"addEdge",2:"removeNode",3:"removeEdge",4:"updateData"}
            idx = max(range(len(btns)), key=lambda i: btns[i])
            currentScenario = possibleScenarios[idx]

            if currentScenario == "addNode":
                self.gStore.add_node(label=self.gStore.currentId)
                return self.gStore.get_elements(), no_update, no_update
            
            elif currentScenario == "addEdge":
                try:
                    source, target = map(int, selectedNodes)
                except ValueError:
                    return no_update
                
                if not self.gStore.G.has_edge(source, target):
                    nl0 = self.gStore.G.nodes[source]["label"]
                    nl1 = self.gStore.G.nodes[target]["label"][-1]
                    self.gStore.add_edge(source,target,label = f"{nl0}{nl1}")
                    return self.gStore.get_elements(), no_update, no_update
                
            elif currentScenario == "removeNode":
                nodeToRemove = int(selectedNodes[-1])
                if nodeToRemove in self.gStore.G.nodes:
                    self.gStore.G.remove_node(nodeToRemove)
                    return self.gStore.get_elements(), [], None
                else:
                    print(f"node {nodeToRemove} is not in G")
                    return no_update, [], None
                
            elif currentScenario == "removeEdge":
                source, target = map(int, selectedNodes)
                if not self.gStore.G.has_edge(source, target):
                    print("edge not in g? how was it selected?")
                else:
                    self.gStore.G.remove_edge(source,target)
                    return self.gStore.get_elements(), [], None
            elif currentScenario == "updateData":
                return self.gStore.get_elements(), no_update, no_update
                
            return no_update, no_update, no_update




        @self.app.callback(
            Output("selected-nodes-display", "children"),
            Input("selectedNodes", "data")
        )
        def display_selected_nodes(selected):
            if not selected:
                return "No nodes selected"
            nodeLabels = [str(self.gStore.G.nodes[int(x)]["label"]) for x in selected ] #if int(x) in self.gStore.G.nodes()
            return f"Selected Nodes: {', '.join(nodeLabels)}"



        @self.app.callback(
            Output("element-properties", "children"),
            Input("selectedElementData", "data") #triggers when a node is selected
        )
        def render_element_edit_form(element):
            if not element:
                return "No element selected"
            kind = element.get("type", "")
            data = element.get("data", {})
            children = [html.H5(f"{kind.capitalize()} Properties")]
            if kind == "node":
                nodeInd = int(data["id"])
                iterable = self.gStore.G.nodes[nodeInd].items()
            elif kind == "edge":
                source,target = map(int,[data["source"],data["target"]])
                iterable = self.gStore.G.edges[source,target].items()

            for key, val in iterable:
                if key == "id":
                    children.append(html.Div([html.Label("id"), html.Div(str(val), style={"fontWeight": "bold"})]))
                    continue
                children.append(
                html.Div([
                    html.Label(key),
                    dcc.Input(
                        id={"prop": "input", "key": key},
                        value=str(val),
                        type="text",
                        debounce=True,
                        style={"width": "100%"},

                    )
                ], style={"marginBottom": "6px"})
            )


            return html.Div(children, style={"border": "1px solid #ccc", "padding": "8px", "borderRadius": "4px"})


        @self.app.callback(
            Output("debug-output", "children", allow_duplicate=True),  # optional debug
            Input({"prop": "input", "key": ALL}, "value"),
            State({"prop": "input", "key": ALL}, "id"),
            State("selectedElementData", "data"),
            prevent_initial_call=True
        )
        def update_element_properties(values, ids, selected):
            if not selected:
                return no_update

            kind = selected.get("type", "")
            data = selected.get("data", {})

            updates = {i["key"]: v for i, v in zip(ids, values)}

            if kind == "node":
                nodeInd = int(data["id"])
                for k, v in updates.items():
                    self.gStore.G.nodes[nodeInd][k] = v

            elif kind == "edge":
                source, target = map(int, [data["source"], data["target"]])
                for k, v in updates.items():
                    self.gStore.G.edges[source, target][k] = v

            return f"Updated {kind}: {updates}"  # optional debug output

        #saving
        @self.app.callback(
            Output("debug-output", "children", allow_duplicate=True),  # optional
            Input("btn-save-network", "n_clicks"),
            State("network-filename", "value"),
            prevent_initial_call=True
        )
        def save_network(nClicks, filename):
            
            try:
                self.gStore.saveToJson(filename)
                return f"Saved network to {filename}"
            except Exception as e:
                return f"Error saving: {str(e)}"

        @self.app.callback(
            Output("cytoCanvas", "elements",allow_duplicate=True),
            Output("networkLoaded", "data", allow_duplicate=True),
            Input("btn-load-network", "n_clicks_timestamp"),
            State("network-filename", "value"),
            prevent_initial_call=True
        )
        def load_network(n_clicks_timestamp, filename):
            try:
                self.gStore.loadFromJson(filename)
                return self.gStore.get_elements(),True
            except Exception as e:
                print(f"Error loading: {e}")
                return no_update
            
        @self.app.callback(
            Output("cytoCanvas", "elements",allow_duplicate=True),
            Output("selectedNodes", "data",allow_duplicate=True),
            Output("selectedElementData", "data",allow_duplicate=True),
            Input("btn-clear-network", "n_clicks_timestamp"),
            prevent_initial_call=True
        )
        def clearNet(n_clicks_timestamp):
            #global self.gStore
            #self.gStore = GraphStore()
            for node in list(self.gStore.G.nodes):
                self.gStore.G.remove_node(node)
            return self.gStore.get_elements(),[],None



        @self.app.callback(
            Output("ic_metadata", "children"),
            Input("btn-save-ic", "n_clicks"),
            State({"type": "ic-input", "key": ALL}, "value"),
            State({"type": "ic-input", "key": ALL}, "id"),
            prevent_initial_call=True
        )
        def save_ic_metadata(nClicks, values, ids):
            updated = {}

            for val, id_ in zip(values, ids):
                key = id_["key"]
                original = self.gStore.metaData[key]
                if isinstance(original, bool):
                    updated[key] = bool(val) and "true" in val  # handle checklist
                elif isinstance(original, float):
                    try: updated[key] = float(val)
                    except: updated[key] = 0.0
                elif isinstance(original, int):
                    try: updated[key] = int(val)
                    except: updated[key] = 0
                elif isinstance(original, (dict, list)):
                    try: updated[key] = json.loads(val)
                    except: updated[key] = original
                else:
                    updated[key] = val

            self.gStore.metaData = updated
            return json.dumps(updated, indent=2)

        @self.app.callback(
        Output("ic-metadata-fields", "children"),
        Input("interval-startup", "n_intervals"),  # or another trigger like app start
        #load network trigger
        #Input("btn-load-network", "n_clicks_timestamp"),
        Input("networkLoaded", "data"),
        prevent_initial_call=False
        )
        def render_ic_metadata_fields(n,n2):
            
            #this  code has to be present in the same scope as the app unfortunately
            meta = self.gStore.metaData
            children = []

            for key, val in meta.items():
                label = html.Label(key)
                inputId = {"type": "ic-input", "key": key}

                if isinstance(val, bool):
                    inputComponent = dcc.Checklist(
                        options=[{"label": "", "value": "true"}],
                        value=["true"] if val else [],
                        id=inputId,
                        style={"marginBottom": "10px"}
                    )
                else:
                    inputComponent = dcc.Input(
                        id=inputId,
                        value=json.dumps(val) if isinstance(val, (dict, list)) else str(val),
                        debounce=True,
                        type="text",
                        style={"width": "100%", "marginBottom": "10px"}
                    )

                children.append(html.Div([label, inputComponent]))

            return children



    def __buildLayout(self):
        self.app.layout = html.Div([
            html.Div([
            html.Button("Add Node", id="btn-add-node", n_clicks_timestamp=0),
            html.Button("Add Edge", id="btn-add-edge", n_clicks_timestamp=0),
            html.Button("Remove Node", id="btn-remove-node", n_clicks_timestamp=0),
            html.Button("Remove Edge", id="btn-remove-edge", n_clicks_timestamp=0),
            html.Div(id="selected-nodes-display", style={"marginTop": "10px"}),
            html.Div(id="debug-output", style={"marginTop": "10px"}),
            ]),
            html.Div([
                cyto.Cytoscape(
                    id="cytoCanvas",
                    layout={"name": "preset"},
                    stylesheet=self.styleSheet,
                    style={"width": "600px", "height": "500px"},
                    elements=[]
                )
            ], style={"display": "inline-block", "verticalAlign": "top"}),
            html.Div([
                html.H4("Filename:"),
                dcc.Input(
                    id="network-filename",
                    type="text",
                    value="test.json",
                    debounce=True,
                    style={"width": "100%", "marginBottom": "6px"}
                ),

                html.Button("Save Network", id="btn-save-network", n_clicks_timestamp=0, style={"marginRight": "10px"}),
                html.Button("Load Network", id="btn-load-network", n_clicks_timestamp=0,style={"marginTop": "5px"}),
                
                html.Hr(),
                html.H4("Selected Element Properties"),
                html.Pre(id="element-properties", style={"whiteSpace": "pre-wrap"}),
                html.Button("Update", id="update-element-props", n_clicks_timestamp=0),

                html.Hr(),
                html.H4("Clear network (irreversible)"),
                html.Button("Clear Network", id="btn-clear-network", n_clicks_timestamp=0),

            ], style={"display": "inline-block", "verticalAlign": "top", "marginLeft": "10px", "width": "230px"}),
            html.Div([
                    html.H4(f"{self.appType} Metadata:"),

                    html.Div(id="ic-metadata-fields"),  # editable fields go here

                    html.Button("Update Metadata", id="btn-save-ic", n_clicks=0, style={"marginTop": "10px"}),

                    html.Pre(id="ic_metadata", style={"whiteSpace": "pre-wrap", "marginTop": "10px"})
            ], style={"display": "inline-block", "verticalAlign": "top", "marginLeft": "10px", "width": "230px"}),

            dcc.Store(id="nodePositions"),
            dcc.Store(id="selectedNodes", data=[]),
            dcc.Store(id="selectedElementData"),
            #dcc.Store(id="dummy-startup-trigger", data={}, storage_type="memory")
            dcc.Interval(id="interval-startup", interval=100, n_intervals=0, max_intervals=1),
            dcc.Store(id="networkLoaded", data=False)

        ], style={"backgroundColor": "white", "padding": "20px"})


