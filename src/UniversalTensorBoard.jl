__precompile__(false)
module UniversalTensorBoard

using TensorFlow

Summaries = TensorFlow.summary

export set_tb_logdir, @tb_log, reset_tb_logs
export tensorboard


mutable struct TensorBoardSession
    logdir::String
    sess::Session
    writer::Summaries.FileWriter
    global_step::Int
end

function TensorBoardSession(logdir; overwrite=false)
    if overwrite
        rm(logdir; force=true, recursive=true)
    end
    mkpath(logdir)
    
    sess= Session(Graph())
    writer = Summaries.FileWriter(logdir; graph=sess.graph)
    TensorBoardSession(realpath(logdir), sess, writer, 0)    
end


TensorBoardSession() = TensorBoardSession("tensorboard_logs", overwrite=true)
# normally the logs don't overwrite, but if you've not given a path, you clearly don't care.

const default_logging_session = Ref(TensorBoardSession())

####################################################################################################
# Interface

"""
    tensorboard()

Opens Tensorboard in the systems default browser.
"""
tensorboard(tb_sess=default_logging_session[]) = visualize(tb_sess.writer)



"""
    set_tb_logdir(logdir, overwrite=false)

Start a new log in the given directory
"""
function set_tb_logdir(logdir, overwrite=false)
    default_logging_session[] = TensorBoardSession(logdir, overwrite=overwrite)
end

"""
    reset_tb_logs()

Reset the current log, deleteing all information
"""
function reset_tb_logs()
    logdir = default_logging_session[].logdir
    default_logging_session[] = TensorBoardSession(logdir, overwrite=true)
end


macro tb_log(name)
    :(_tb_log($(esc(string(name))), $(esc(name))))
end


#################################################################
# Implmentation 
function _tb_log(name, value; tb_sess = default_logging_session[])
    placeholder_node, summary_node  = get_nodes!(tb_sess, name, value)
    summaries = run(tb_sess.sess, summary_node,  Dict(placeholder_node=>value))
    write(tb_sess.writer, summaries, tb_sess.global_step)
    tb_sess.global_step+=1
    value
end

name_fix(name)=replace(name, r"[^A-Za-z0-9.]|\s", "_")
placeholder_node_name(name) = name_fix(name *"_input")
summary_node_name(name) = name_fix(name)

has_node(graph, name) = !isnull(TensorFlow.get_node_by_name(graph, name))

function get_nodes!(tb_sess, name, value)
    
    gg = tb_sess.sess.graph 
    if has_node(gg, placeholder_node_name(name)) 
        (gg[placeholder_node_name(name)], gg[summary_node_name(name)])
    else
        TensorFlow.as_default(gg) do
            create_nodes!(name, value)
        end
    end
end


function create_nodes!(name, value::T) where T<:Number
    placeholder_node = placeholder(T; name=placeholder_node_name(name), shape=[])
    summary_node = Summaries.scalar(name, placeholder_node; name=summary_node_name(name))
    (placeholder_node, summary_node)
end

function create_nodes!(name, value::AbstractArray{T}) where T<:Number
    shape = collect(size(value))
    placeholder_node=placeholder(T; name=placeholder_node_name(name), shape=shape)
    summary_node = Summaries.histogram(name, placeholder_node; name=summary_node_name(name))
    (placeholder_node, summary_node)
end


end # modulee
