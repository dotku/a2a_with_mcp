from enum import Enum
import json
from typing import Sequence
import redditwarp.SYNC
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from mcp.shared.exceptions import McpError
from pydantic import BaseModel
import logging
import os
import asyncio
import threading
import uvicorn
from fastapi import FastAPI, Request, Response
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# --- Configure Logging --- 
log_file = os.path.join(os.path.dirname(__file__), 'mcp_server.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        # Add StreamHandler to help with debugging
        logging.StreamHandler() 
    ]
)
logger = logging.getLogger(__name__)
# ---

# Create FastAPI app for HTTP server
app = FastAPI()

# Global server instance
reddit_server_instance = None

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint to verify server is running"""
    logger.info("Health check endpoint called")
    return {"status": "ok"}

# RPC endpoint for tool calls
@app.post("/")
async def handle_rpc(request: Request):
    """Handle JSON-RPC requests"""
    global reddit_server_instance
    
    if not reddit_server_instance:
        logger.error("Reddit server instance not initialized")
        return Response(
            content=json.dumps({"error": {"code": -32603, "message": "Server not initialized"}}),
            media_type="application/json",
            status_code=500
        )
    
    try:
        data = await request.json()
        logger.info(f"Received JSON-RPC request: {data}")
        
        # Validate JSON-RPC structure
        if not isinstance(data, dict):
            return Response(
                content=json.dumps({"error": {"code": -32600, "message": "Invalid Request"}}),
                media_type="application/json",
                status_code=400
            )
        
        method = data.get("method")
        if method != "call_tool":
            return Response(
                content=json.dumps({"error": {"code": -32601, "message": "Method not found"}}),
                media_type="application/json",
                status_code=404
            )
        
        params = data.get("params", {})
        if not isinstance(params, dict):
            return Response(
                content=json.dumps({"error": {"code": -32602, "message": "Invalid params"}}),
                media_type="application/json",
                status_code=400
            )
        
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        # Call the appropriate Reddit tool
        logger.info(f"Calling tool '{tool_name}' with arguments: {arguments}")
        
        if tool_name == RedditTools.GET_SUBREDDIT_NEW_POSTS.value:
            subreddit_name = arguments.get('subreddit_name')
            if not subreddit_name:
                return Response(
                    content=json.dumps({"error": {"code": -32602, "message": "Missing required parameter: subreddit_name"}}),
                    media_type="application/json",
                    status_code=400
                )
            
            limit = arguments.get('limit', 10)
            session_id = arguments.get('session_id', "")
            
            try:
                result = reddit_server_instance.get_subreddit_new_posts(
                    subreddit_name=subreddit_name,
                    limit=limit,
                    session_id=session_id
                )
                # Format the result as expected by the client
                text_contents = [post.model_dump_json() for post in result]
                return Response(
                    content=json.dumps({
                        "jsonrpc": "2.0",
                        "id": data.get("id"),
                        "result": [{"type": "text", "text": content} for content in text_contents]
                    }),
                    media_type="application/json"
                )
            except Exception as e:
                logger.exception(f"Error executing tool: {e}")
                return Response(
                    content=json.dumps({
                        "jsonrpc": "2.0",
                        "id": data.get("id"),
                        "error": {"code": -32603, "message": str(e)}
                    }),
                    media_type="application/json",
                    status_code=500
                )
        else:
            return Response(
                content=json.dumps({"error": {"code": -32601, "message": f"Tool not found: {tool_name}"}}),
                media_type="application/json",
                status_code=404
            )
    
    except Exception as e:
        logger.exception(f"Error processing request: {e}")
        return Response(
            content=json.dumps({"error": {"code": -32603, "message": str(e)}}),
            media_type="application/json",
            status_code=500
        )

# Function to start HTTP server
def start_http_server(port=10101):
    """Start the HTTP server on the specified port"""
    logger.info(f"Starting HTTP server on port {port}")
    # Use a more robust configuration for uvicorn
    try:
        # Create a new event loop for this thread
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Now run uvicorn with the new event loop
        uvicorn.run(
            app, 
            host="127.0.0.1",  # More restrictive binding
            port=port, 
            log_level="info",
            access_log=True,
            timeout_keep_alive=120
        )
    except Exception as e:
        logger.error(f"Error starting HTTP server: {e}")
        raise

class PostType(str, Enum):
    LINK = "link"
    TEXT = "text"
    GALLERY = "gallery"
    UNKNOWN = "unknown"


class RedditTools(str, Enum):
    GET_FRONTPAGE_POSTS = "get_frontpage_posts"
    GET_SUBREDDIT_INFO = "get_subreddit_info"
    GET_SUBREDDIT_HOT_POSTS = "get_subreddit_hot_posts"
    GET_SUBREDDIT_NEW_POSTS = "get_subreddit_new_posts"
    GET_SUBREDDIT_TOP_POSTS = "get_subreddit_top_posts"
    GET_SUBREDDIT_RISING_POSTS = "get_subreddit_rising_posts"
    GET_POST_CONTENT = "get_post_content"
    GET_POST_COMMENTS = "get_post_comments"


class SubredditInfo(BaseModel):
    name: str
    subscriber_count: int
    description: str | None


class Post(BaseModel):
    id: str
    title: str
    author: str
    score: int
    subreddit: str
    url: str
    created_at: str
    comment_count: int
    post_type: PostType
    content: str | None


class Comment(BaseModel):
    id: str
    author: str
    body: str
    score: int
    replies: list['Comment'] = []


class Moderator(BaseModel):
    name: str


class PostDetail(BaseModel):
    post: Post
    comments: list[Comment]


class RedditServer:
    def __init__(self):
        logger.info("Initializing RedditServer and redditwarp client.")
        try:
            self.client = redditwarp.SYNC.Client()
            logger.info("Redditwarp client initialized successfully.")
        except Exception as e:
            logger.exception("Failed to initialize redditwarp client")
            raise

    def _get_post_type(self, submission) -> PostType:
        """Helper method to determine post type"""
        if isinstance(submission, redditwarp.models.submission_SYNC.LinkPost):
            return PostType.LINK
        elif isinstance(submission, redditwarp.models.submission_SYNC.TextPost):
            return PostType.TEXT
        elif isinstance(submission, redditwarp.models.submission_SYNC.GalleryPost):
            return PostType.GALLERY
        return PostType.UNKNOWN

    # The type can actually be determined by submission.post_hint
    # - self for text
    # - image for image
    # - hosted:video for video
    def _get_post_content(self, submission) -> str | None:
        """Helper method to extract post content based on type"""
        if isinstance(submission, redditwarp.models.submission_SYNC.LinkPost):
            return submission.permalink
        elif isinstance(submission, redditwarp.models.submission_SYNC.TextPost):
            return submission.body
        elif isinstance(submission, redditwarp.models.submission_SYNC.GalleryPost):
            return str(submission.gallery_link)
        return None

    def _build_post(self, submission) -> Post:
        """Helper method to build Post object from submission"""
        return Post(
            id=submission.id36,
            title=submission.title,
            author=submission.author_display_name or '[deleted]',
            score=submission.score,
            subreddit=submission.subreddit.name,
            url=submission.permalink,
            created_at=submission.created_at.astimezone().isoformat(),
            comment_count=submission.comment_count,
            post_type=self._get_post_type(submission),
            content=self._get_post_content(submission)
        )

    def get_frontpage_posts(self, limit: int = 10) -> list[Post]:
        """Get hot posts from Reddit frontpage"""
        posts = []
        for subm in self.client.p.front.pull.hot(limit):
            posts.append(self._build_post(subm))
        return posts

    def get_subreddit_info(self, subreddit_name: str) -> SubredditInfo:
        """Get information about a subreddit"""
        # Clean up subreddit name
        subreddit_name = self._clean_subreddit_name(subreddit_name)
        logger.info(f"Executing get_subreddit_info for r/{subreddit_name}")
        
        subr = self.client.p.subreddit.fetch_by_name(subreddit_name)
        return SubredditInfo(
            name=subr.name,
            subscriber_count=subr.subscriber_count,
            description=subr.public_description
        )

    def _build_comment_tree(self, node, depth: int = 3) -> Comment | None:
        """Helper method to recursively build comment tree"""
        if depth <= 0 or not node:
            return None

        comment = node.value
        replies = []
        for child in node.children:
            child_comment = self._build_comment_tree(child, depth - 1)
            if child_comment:
                replies.append(child_comment)

        return Comment(
            id=comment.id36,
            author=comment.author_display_name or '[deleted]',
            body=comment.body,
            score=comment.score,
            replies=replies
        )

    def get_subreddit_hot_posts(self, subreddit_name: str, limit: int = 10) -> list[Post]:
        """Get hot posts from a specific subreddit"""
        # Clean up subreddit name
        subreddit_name = self._clean_subreddit_name(subreddit_name)
        logger.info(f"Executing get_subreddit_hot_posts for r/{subreddit_name}, limit={limit}")
        
        posts = []
        for subm in self.client.p.subreddit.pull.hot(subreddit_name, limit):
            posts.append(self._build_post(subm))
        return posts

    def _clean_subreddit_name(self, subreddit_name: str) -> str:
        """Clean subreddit name by removing r/ prefix if present."""
        if not subreddit_name:
            return ""
            
        # Remove 'r/' prefix if present
        if subreddit_name.startswith('r/'):
            cleaned_name = subreddit_name[2:]
            logger.info(f"Removed 'r/' prefix from subreddit name: now using '{cleaned_name}'")
            return cleaned_name
        return subreddit_name

    def get_subreddit_new_posts(self, subreddit_name: str, limit: int = 10, session_id: str = "") -> list[Post]:
        """Get new posts from a specific subreddit"""
        logger.info(f"Executing get_subreddit_new_posts for r/{subreddit_name}, limit={limit}, session_id={session_id}")
        
        # Clean up subreddit name
        subreddit_name = self._clean_subreddit_name(subreddit_name)
        
        # Add more detailed logging
        logger.info(f"Using cleaned subreddit name: '{subreddit_name}'")
        
        # Handle empty subreddit name
        if not subreddit_name:
            logger.error("Empty subreddit name after cleaning")
            return []
            
        posts = []
        try:
            # Add logging around the actual API call
            logger.info(f"Calling redditwarp: client.p.subreddit.pull.new(subreddit='{subreddit_name}', limit={limit})")
            try:
                iterator = self.client.p.subreddit.pull.new(subreddit_name, limit)
                for i, subm in enumerate(iterator):
                    logger.debug(f"Processing post {i+1} from r/{subreddit_name}")
                    posts.append(self._build_post(subm))
                logger.info(f"Successfully fetched {len(posts)} new posts from r/{subreddit_name}")
            except Exception as api_err:
                # Catch Reddit API specific errors
                logger.exception(f"Reddit API error for subreddit '{subreddit_name}': {str(api_err)}")
                # Return empty list instead of raising to avoid breaking the sentiment analysis
                return []
                
            return posts
        except Exception as e:
            logger.exception(f"Error fetching new posts from r/{subreddit_name}: {str(e)}")
            # Return an empty list rather than raising to prevent tool failure
            return []

    def get_subreddit_top_posts(self, subreddit_name: str, limit: int = 10, time: str = '') -> list[Post]:
        """Get top posts from a specific subreddit"""
        # Clean up subreddit name
        subreddit_name = self._clean_subreddit_name(subreddit_name)
        logger.info(f"Executing get_subreddit_top_posts for r/{subreddit_name}, limit={limit}, time={time}")
        
        posts = []
        for subm in self.client.p.subreddit.pull.top(subreddit_name, limit, time=time):
            posts.append(self._build_post(subm))
        return posts

    def get_subreddit_rising_posts(self, subreddit_name: str, limit: int = 10) -> list[Post]:
        """Get rising posts from a specific subreddit"""
        # Clean up subreddit name
        subreddit_name = self._clean_subreddit_name(subreddit_name)
        logger.info(f"Executing get_subreddit_rising_posts for r/{subreddit_name}, limit={limit}")
        
        posts = []
        for subm in self.client.p.subreddit.pull.rising(subreddit_name, limit):
            posts.append(self._build_post(subm))
        return posts

    def get_post_content(self, post_id: str, comment_limit: int = 10, comment_depth: int = 3) -> PostDetail:
        """Get detailed content of a specific post including comments"""
        submission = self.client.p.submission.fetch(post_id)
        post = self._build_post(submission)

        # Fetch comments
        comments = self.get_post_comments(post_id, comment_limit)
        
        return PostDetail(post=post, comments=comments)

    def get_post_comments(self, post_id: str, limit: int = 10) -> list[Comment]:
        """Get comments from a post"""
        logger.info(f"Executing get_post_comments for post_id={post_id}, limit={limit}")
        comments = []
        try:
            logger.info(f"Calling redditwarp: client.p.comment_tree.fetch(post_id='{post_id}', limit={limit})")
            tree_node = self.client.p.comment_tree.fetch(post_id, sort='top', limit=limit)
            for i, node in enumerate(tree_node.children):
                logger.debug(f"Processing comment node {i+1} for post {post_id}")
                comment = self._build_comment_tree(node)
                if comment:
                    comments.append(comment)
            logger.info(f"Finished fetching {len(comments)} comments for post {post_id}")
            return comments
        except Exception as e:
            logger.exception(f"Error fetching comments for post {post_id}")
            raise


async def serve(port=10101) -> None:
    logger.info("Starting MCP server setup.")
    server = Server("mcp-reddit")
    
    # Create server instance and store globally for HTTP server to access
    global reddit_server_instance
    reddit_server_instance = RedditServer()
    logger.info("RedditServer instance created.")
    
    # Try a simpler approach - run the HTTP server on a named port and test it
    try:
        # Import uvicorn config to run programmatically
        import uvicorn
        from uvicorn import Config
        import multiprocessing
        
        # Create a new process for the HTTP server instead of a thread
        server_process = multiprocessing.Process(
            target=uvicorn.run,
            kwargs={
                "app": app,
                "host": "127.0.0.1", 
                "port": port,
                "log_level": "info"
            }
        )
        server_process.daemon = True
        server_process.start()
        logger.info(f"Started HTTP server in separate process (PID: {server_process.pid})")
        
        # Give the server a moment to start
        await asyncio.sleep(2)
        
        # Try to connect to check it's actually running
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        
        if result == 0:
            logger.info(f"HTTP server is running on port {port}")
        else:
            logger.error(f"HTTP server is not responding on port {port}")
    except Exception as e:
        logger.exception(f"Error starting HTTP server process: {e}")
    
    # Continue with the stdio server setup
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available Reddit tools."""
        logger.info("list_tools endpoint called.")
        return [
            Tool(
                name=RedditTools.GET_FRONTPAGE_POSTS.value,
                description="Get recent posts from the Reddit frontpage.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of posts to return (default: 10)",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100
                        }
                    },
                },
            ),
            Tool(
                name=RedditTools.GET_SUBREDDIT_INFO.value,
                description="Get information about a specific subreddit, including subscriber count.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "subreddit_name": {
                            "type": "string",
                            "description": "Name of the subreddit without the 'r/' prefix",
                        }
                    },
                    "required": ["subreddit_name"]
                }
            ),
            Tool(
                name=RedditTools.GET_SUBREDDIT_HOT_POSTS.value,
                description="Get the hot posts in a subreddit.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "subreddit_name": {
                            "type": "string",
                            "description": "Name of the subreddit without the 'r/' prefix",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of posts to return (default: 10)",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100
                        }
                    },
                    "required": ["subreddit_name"]
                }
            ),
            Tool(
                name=RedditTools.GET_SUBREDDIT_NEW_POSTS.value,
                description="Get the newest posts in a subreddit.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "subreddit_name": {
                            "type": "string",
                            "description": "Name of the subreddit without the 'r/' prefix",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of posts to return (default: 10)",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Optional session identifier for tracking purposes",
                            "default": ""
                        }
                    },
                    "required": ["subreddit_name"]
                }
            ),
            Tool(
                name=RedditTools.GET_SUBREDDIT_TOP_POSTS.value,
                description="Get top posts from a specific subreddit",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "subreddit_name": {
                            "type": "string",
                            "description": "Name of the subreddit (e.g. 'Python', 'news')",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of posts to return (default: 10)",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100
                        },
                        "time": {
                            "type": "string",
                            "description": "Time filter for top posts (e.g. 'hour', 'day', 'week', 'month', 'year', 'all')",
                            "default": "",
                            "enum": ["", "hour", "day", "week", "month", "year", "all"]
                        }
                    },
                    "required": ["subreddit_name"]
                }
            ),
            Tool(
                name=RedditTools.GET_SUBREDDIT_RISING_POSTS.value,
                description="Get rising posts from a specific subreddit",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "subreddit_name": {
                            "type": "string",
                            "description": "Name of the subreddit (e.g. 'Python', 'news')",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of posts to return (default: 10)",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100
                        }
                    },
                    "required": ["subreddit_name"]
                }
            ),
            Tool(
                name=RedditTools.GET_POST_CONTENT.value,
                description="Get detailed content of a specific post",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "post_id": {
                            "type": "string",
                            "description": "ID of the post",
                        },
                        "comment_limit": {
                            "type": "integer",
                            "description": "Number of top-level comments to return (default: 10)",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100
                        },
                        "comment_depth": {
                            "type": "integer",
                            "description": "Maximum depth of comment tree (default: 3)",
                            "default": 3,
                            "minimum": 1,
                            "maximum": 10
                        }
                    },
                    "required": ["post_id"]
                }
            ),
            Tool(
                name=RedditTools.GET_POST_COMMENTS.value,
                description="Get comments from a post",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "post_id": {
                            "type": "string",
                            "description": "ID of the post",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of comments to return (default: 10)",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100
                        }
                    },
                    "required": ["post_id"]
                }
            ),
        ]

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        """Call the appropriate Reddit tool based on the name."""
        logger.info(f"call_tool endpoint called: name='{name}', args={arguments}")
        try:
            if name == RedditTools.GET_SUBREDDIT_NEW_POSTS.value:
                # Parameter validation with detailed error messages
                subreddit_name = arguments.get('subreddit_name')
                if not subreddit_name:
                    logger.error("Missing required parameter: subreddit_name")
                    return [TextContent(type="text", text=json.dumps({"tool_error": "Missing required parameter: subreddit_name"}))]
                
                if not isinstance(subreddit_name, str) or len(subreddit_name.strip()) == 0:
                    logger.error(f"Invalid subreddit_name parameter: {subreddit_name}")
                    return [TextContent(type="text", text=json.dumps({"tool_error": "subreddit_name must be a non-empty string"}))]
                
                # Clean subreddit name
                subreddit_name = reddit_server_instance._clean_subreddit_name(subreddit_name)
                
                limit = arguments.get('limit', 10)
                # Validate limit is a positive integer
                if not isinstance(limit, int) or limit <= 0 or limit > 100:
                    logger.warning(f"Invalid limit parameter: {limit}, using default of 10")
                    limit = 10
                
                # Get session_id if provided (optional parameter)
                session_id = arguments.get('session_id', "")
                logger.info(f"Using session_id: {session_id if session_id else 'default (empty)'}")
                
                try:
                    result = reddit_server_instance.get_subreddit_new_posts(
                        subreddit_name=subreddit_name,
                        limit=limit,
                        session_id=session_id
                    )
                    logger.info(f"Tool '{name}' executed. Result count: {len(result)}")
                    # Return each post as a separate TextContent JSON string
                    return [TextContent(type="text", text=post.model_dump_json()) for post in result]
                except Exception as e:
                    logger.exception(f"Error executing get_subreddit_new_posts for {subreddit_name}: {str(e)}")
                    return [TextContent(type="text", text=json.dumps({"tool_error": f"Failed to fetch posts from r/{subreddit_name}: {str(e)}"}))]
                
            elif name == RedditTools.GET_POST_COMMENTS.value:
                # Parameter validation 
                post_id = arguments.get('post_id')
                if not post_id:
                    logger.error("Missing required parameter: post_id")
                    return [TextContent(type="text", text=json.dumps({"tool_error": "Missing required parameter: post_id"}))]
                
                limit = arguments.get('limit', 10)
                if not isinstance(limit, int) or limit <= 0 or limit > 100:
                    logger.warning(f"Invalid limit parameter: {limit}, using default of 10")
                    limit = 10
                    
                try:
                    result = reddit_server_instance.get_post_comments(
                        post_id=post_id,
                        limit=limit
                    )
                    logger.info(f"Tool '{name}' executed. Result count: {len(result)}")
                    return [TextContent(type="text", text=comment.model_dump_json()) for comment in result]
                except Exception as e:
                    logger.exception(f"Error executing get_post_comments for {post_id}: {str(e)}")
                    return [TextContent(type="text", text=json.dumps({"tool_error": f"Failed to fetch comments for post {post_id}: {str(e)}"}))]
                
            elif name == RedditTools.GET_FRONTPAGE_POSTS.value:
                limit = arguments.get("limit", 10)
                if not isinstance(limit, int) or limit <= 0 or limit > 100:
                    logger.warning(f"Invalid limit parameter: {limit}, using default of 10")
                    limit = 10
                    
                try:
                    result = reddit_server_instance.get_frontpage_posts(limit)
                    logger.info(f"Tool '{name}' executed. Result count: {len(result)}")
                    return [TextContent(type="text", text=post.model_dump_json()) for post in result]
                except Exception as e:
                    logger.exception(f"Error executing get_frontpage_posts: {str(e)}")
                    return [TextContent(type="text", text=json.dumps({"tool_error": f"Failed to fetch frontpage posts: {str(e)}"}))]

            elif name == RedditTools.GET_SUBREDDIT_INFO.value:
                subreddit_name = arguments.get("subreddit_name")
                if not subreddit_name:
                    logger.error("Missing required parameter: subreddit_name")
                    return [TextContent(type="text", text=json.dumps({"tool_error": "Missing required parameter: subreddit_name"}))]
                
                # Clean subreddit name
                subreddit_name = reddit_server_instance._clean_subreddit_name(subreddit_name)
                
                try:
                    result = reddit_server_instance.get_subreddit_info(subreddit_name)
                    logger.info(f"Tool '{name}' executed successfully.")
                    return [TextContent(type="text", text=result.model_dump_json())]
                except Exception as e:
                    logger.exception(f"Error executing get_subreddit_info for {subreddit_name}: {str(e)}")
                    return [TextContent(type="text", text=json.dumps({"tool_error": f"Failed to fetch subreddit info for r/{subreddit_name}: {str(e)}"}))]

            elif name == RedditTools.GET_SUBREDDIT_HOT_POSTS.value:
                subreddit_name = arguments.get("subreddit_name")
                if not subreddit_name:
                    logger.error("Missing required parameter: subreddit_name")
                    return [TextContent(type="text", text=json.dumps({"tool_error": "Missing required parameter: subreddit_name"}))]
                
                # Clean subreddit name
                subreddit_name = reddit_server_instance._clean_subreddit_name(subreddit_name)
                
                limit = arguments.get("limit", 10)
                if not isinstance(limit, int) or limit <= 0 or limit > 100:
                    logger.warning(f"Invalid limit parameter: {limit}, using default of 10")
                    limit = 10
                
                try:
                    result = reddit_server_instance.get_subreddit_hot_posts(subreddit_name, limit)
                    logger.info(f"Tool '{name}' executed. Result count: {len(result)}")
                    return [TextContent(type="text", text=post.model_dump_json()) for post in result]
                except Exception as e:
                    logger.exception(f"Error executing get_subreddit_hot_posts for {subreddit_name}: {str(e)}")
                    return [TextContent(type="text", text=json.dumps({"tool_error": f"Failed to fetch hot posts from r/{subreddit_name}: {str(e)}"}))]

            elif name == RedditTools.GET_SUBREDDIT_TOP_POSTS.value:
                subreddit_name = arguments.get("subreddit_name")
                if not subreddit_name:
                    logger.error("Missing required parameter: subreddit_name")
                    return [TextContent(type="text", text=json.dumps({"tool_error": "Missing required parameter: subreddit_name"}))]
                
                # Clean subreddit name
                subreddit_name = reddit_server_instance._clean_subreddit_name(subreddit_name)
                
                limit = arguments.get("limit", 10)
                if not isinstance(limit, int) or limit <= 0 or limit > 100:
                    logger.warning(f"Invalid limit parameter: {limit}, using default of 10")
                    limit = 10
                    
                time = arguments.get("time", "")
                
                try:
                    result = reddit_server_instance.get_subreddit_top_posts(subreddit_name, limit, time)
                    logger.info(f"Tool '{name}' executed. Result count: {len(result)}")
                    return [TextContent(type="text", text=post.model_dump_json()) for post in result]
                except Exception as e:
                    logger.exception(f"Error executing get_subreddit_top_posts for {subreddit_name}: {str(e)}")
                    return [TextContent(type="text", text=json.dumps({"tool_error": f"Failed to fetch top posts from r/{subreddit_name}: {str(e)}"}))]

            elif name == RedditTools.GET_SUBREDDIT_RISING_POSTS.value:
                subreddit_name = arguments.get("subreddit_name")
                if not subreddit_name:
                    logger.error("Missing required parameter: subreddit_name")
                    return [TextContent(type="text", text=json.dumps({"tool_error": "Missing required parameter: subreddit_name"}))]
                
                # Clean subreddit name
                subreddit_name = reddit_server_instance._clean_subreddit_name(subreddit_name)
                
                limit = arguments.get("limit", 10)
                if not isinstance(limit, int) or limit <= 0 or limit > 100:
                    logger.warning(f"Invalid limit parameter: {limit}, using default of 10")
                    limit = 10
                
                try:
                    result = reddit_server_instance.get_subreddit_rising_posts(subreddit_name, limit)
                    logger.info(f"Tool '{name}' executed. Result count: {len(result)}")
                    return [TextContent(type="text", text=post.model_dump_json()) for post in result]
                except Exception as e:
                    logger.exception(f"Error executing get_subreddit_rising_posts for {subreddit_name}: {str(e)}")
                    return [TextContent(type="text", text=json.dumps({"tool_error": f"Failed to fetch rising posts from r/{subreddit_name}: {str(e)}"}))]

            elif name == RedditTools.GET_POST_CONTENT.value:
                post_id = arguments.get("post_id")
                if not post_id:
                    logger.error("Missing required parameter: post_id")
                    return [TextContent(type="text", text=json.dumps({"tool_error": "Missing required parameter: post_id"}))]
                
                comment_limit = arguments.get("comment_limit", 10)
                if not isinstance(comment_limit, int) or comment_limit <= 0 or comment_limit > 100:
                    logger.warning(f"Invalid comment_limit parameter: {comment_limit}, using default of 10")
                    comment_limit = 10
                    
                comment_depth = arguments.get("comment_depth", 3)
                if not isinstance(comment_depth, int) or comment_depth <= 0 or comment_depth > 10:
                    logger.warning(f"Invalid comment_depth parameter: {comment_depth}, using default of 3")
                    comment_depth = 3
                
                try:
                    result = reddit_server_instance.get_post_content(post_id, comment_limit, comment_depth)
                    logger.info(f"Tool '{name}' executed successfully.")
                    return [TextContent(type="text", text=result.model_dump_json())]
                except Exception as e:
                    logger.exception(f"Error executing get_post_content for {post_id}: {str(e)}")
                    return [TextContent(type="text", text=json.dumps({"tool_error": f"Failed to fetch post content for {post_id}: {str(e)}"}))]

            else:
                logger.warning(f"Unknown tool name received: {name}")
                return [TextContent(type="text", text=json.dumps({"tool_error": f"Unknown tool: {name}"}))]

        except Exception as e:
            logger.exception(f"Error executing tool '{name}' with args {arguments}")
            return [TextContent(type="text", text=json.dumps({"tool_error": str(e)}))]

    logger.info("Starting stdio_server.")
    # Create initialization options first
    options = server.create_initialization_options()
    # Then use stdio_server properly
    async with stdio_server() as (read_stream, write_stream):
        # Pass the options as the third argument to run()
        await server.run(read_stream, write_stream, options)
        
    logger.info("stdio_server finished.")
