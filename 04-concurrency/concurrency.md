#Exceptions 

- `undefined` and `error` can "return" any type
- This is because they don't actually return anything but throw *language level* exceptions 
- To use exceptions directly __import Control.Exception__ 

~~~{.haskell}
import Prelude hiding (catch) -- hiding prevents import of specific symbols 
import Control.Exception 
~~~ 

- `Control.Exception` gives access to `throw`, `throwIO`, `catch` 
- Prelude has an older implementation of catch that needs to be ignored. 

#Simple Example 

~~~{.haskell}
{-# LANGUAGE DeriveDataTypeable #-}

import Prelude hiding (catch)
import Control.Exception
import Data.Typeable

data MyError = MyError String deriving (Show, Typeable)
instance Exception MyError

catcher :: IO a -> IO (Maybe a)
catcher action = fmap Just action `catch` handler
    where handler (MyError msg) = do putStrLn msg; return Nothing
~~~

~~~
*Main> catcher $ readFile "/dev/null"
Just ""
*Main> catcher $ throwIO $ MyError "something bad"
something bad
Nothing
~~~

- `catch` takes as its arguments an `action` and a `handler`
~~~{.haskell}
catch :: (Exception e) => IO a -> (e -> IO a) -> IO a
~~~

- It __executes__ the given __action__ first and if that throws an `Exception` then the handler is called
- The above `ghci` snippet captures this. 

####Some notes from the code above 
- `DerivedDataTypeable` pragma used will be dicussed in a later lecture 
- `Typeable` means run time type info for exceptions `(Discuss this with DM)` 
- can create your own instance of Exception 
- need to __specify the constructor explicitly__ as shown ; `otherwise` it is a `compiler error`.
- `SomeException` is used to  _catch all exceptions_. 
- Constructor pattern `e@(SomeException _)` catches all exceptions
- type argument to SomeException has to be instance of type Exception 
- `toException` and `fromException` two method on Exception typeclass. 
- `toException` converts e to SomeException 
- `fromException` returns either Nothing or Just _ 
- take away message : use __@(SomeException _ ) to catch anything__

#Exceptions in Pure Code
- can throw exceptions in `Pure` code but can only catch them in `IO`
	+ Because evaluation order depends on implementation 
	+ Which error is thrown by `(error "one") + (error "two")` is non-deterministic
- Use _throw_ only when *throwIO* is not possible 

#Exceptions and Laziness 
~~~{.haskell}
pureCatcher :: a -> IO (Maybe a)
pureCatcher a = (a `seq` return (Just a))
                `catch` \(SomeException _) -> return Nothing
~~~

- Consider the following inputs to `pureCatcher` 

~~~
*Main> pureCatcher $ 1 `div` 0
Nothing
*Main> pureCatcher (undefined :: String)
Nothing
~~~

- This is the interesting case: 

~~~
*Main> pureCatcher (undefined:undefined :: String)
Just "*** Exception: Prelude.undefined
~~~

- *Why does this happen*
	+ __catch only catches expceptions when thunks are evaluated__
	+ __Evaluating the list does not evaluate the head or tail__
	+ Only the constructor `(:)` or `[]` is evaluated as shown below
~~~
*Main> seq (undefined:undefined) ()
()
~~~
- function called `deepseq`  in library of __same name__ that _traverses an entire data structure_. evaluating it completely before returning the second argument, just like its namesake `seq` 
#### A Few More Exception Functions
~~~{.haskell}
try :: Exception e => IO a -> IO (Either e a)
~~~
- `try` returns `Right a` _normally_ and `Left e` if an exceptions occurred. 
- `finally` and `onException` run a clean up action 
- `catchJust` catches only exceptions _matching_ a `predicate on value`

#Monadic Exceptions 
- Language level exceptions can be cumbersome for non-IO actions
	+ Non-determinism is annoying  
	+ Many monads built on top of IO can't catch such exceptions 
- Implement error handling in the Monad. 
	+ In the `Maybe Monad` _Nothing_ can be used to indicate _failure_ 

#Haskell Threads  
- Haskell provides __user-level__ threads in `Control.Concurrent` 
- `forkIO` - creates new thread 
- Threads are very lightweight, switching between threads does not require any intervention from the OS 
- reading a socket does not put whole program to sleep just the  particular thread
- `throwTo` allows you raise an exception in *ANOTHER* thread. 
- `killThread` is a special case of _throwTo_

#Example: timeOut 

~~~{.haskell} 
data TimedOut = TimedOut UTCTime deriving (Eq, Show, Typeable)
instance Exception TimedOut

timeout :: Int -> IO a -> IO (Maybe a)
timeout usec action = do
  -- Create unique exception val (for nested timeouts):
  expired <- fmap TimedOut getCurrentTime

  ptid <- myThreadId
  let child = do threadDelay usec
                 throwTo ptid expired
      parent = do ctid <- forkIO child
                  result <- action
                  killThread ctid
                  return $ Just result
  catchJust (\e -> if e == expired then Just e else Nothing) 
            parent
            (\_ -> return Nothing)
~~~

- if parent's computation is not over within `usec` then the child throws an Exception
- Store current time in our exception and use `catchJust` to make sure we do not handle some other nested exception
- `System.Timeout` has a slightly better version of the same function 

# MVars
- MVar = `M`utable `Var`iable
- The __MVar type__ lets threads communicate via `mutable shared variables`
- `MVar t` is mutable var of `type t`. It is either _full_ of _empty_ 
- MVar is like a channel of depth one ( Go analogy )
- `tryTakeMVar` and `tryPutMVar` will return immediately ( Non-blocking).
- takeMVar and putMVar are blocking versions. 
- If an Mvar is `empty` putMVar will fill it with the supplied value 
- If an Mvar is `full` takeMVar makes it _empty_ and returns the former value
- MVar is a value on heap ; when references go away it is garbage collected. 
- MVars are built into run time 
- Under the covers there is a mutex protecting it. 

#Example: Pingpong Benchmark
- context switch n times between two threads 
- child and parent call themselves recurscively 
- child reduces the recursion to ensure the parent does not call itself recurscively forever 
~~~{.haskell}
import Control.Concurrent
import Control.Exception
import Control.Monad

pingpong :: Bool -> Int -> IO ()
pingpong v n = do
  mvc <- newEmptyMVar   -- MVar read by child
  mvp <- newEmptyMVar   -- MVar read by parent
  let parent n | n > 0 = do when v $ putStr $ " " ++ show n
                            putMVar mvc n
                            takeMVar mvp >>= parent
               | otherwise = return ()
      child = do n <- takeMVar mvc
                 putMVar mvp (n - 1)
                 child
  tid <- forkIO child
  parent n `finally` killThread tid
  when v $ putStrLn ""
~~~
- We will benchmark Pingpong with [__criterion__.](http://hackage.haskell.org/package/criterion) A benchmarking lib written by Bryan
~~~{.haskell}
import Criterion.Main

...

main :: IO ()
main = defaultMain [
        bench "thread switch test" mybench
       ]
    where mybench = pingpong False 10000
~~~

- __Takes 3.8ms for 20K thread switches ~190nsec/switch__
	+ Proof that haskell threads are lightweight 

#OS Threads + Bound vs Unbound Threads 

~~~{.haskell} 
forkOS :: IO () -> IO ThreadId
~~~

- `forkOS` creates a Haskell thread _bound_ to a new OS thread 
- When program is linked with __-threaded__ initial thread is bound. 

~~~
$ ghc -threaded -O pingpong.hs 
Linking pingpong ...
$ ./pingpong
...
mean: 121.1729 ms, lb 120.5601 ms, ub 121.7044 ms, ci 0.950
...
~~~

- __Why is it so slow ???__ 
	+ Without __-threaded__ all Haskell threads run in _one OS thread_ 
	+ In such a situation *switching between threads* is just a `function call`
	+ With __-threaded__ initial thread is _bound_
	+ We ended up context switching linux thread and a haskell thread 
	+ Therefore we were actually benchmarking Linux
	+ Solve this by wrapping initial thread in `forkIO` to make it _unbounded_
	+ Can do the wrapping yourself or use built in library function [__runInUnboundThread__](http://hackage.haskell.org/package/base-4.7.0.0/docs/Control-Concurrent.html#v:runInUnboundThread)

#### What Good are OS Threads 

- If __unbound thread__ blocks it can block __whole program__ 
- FFI functions may expect to be called from same thread - so need bounded threads there. 
- With _-threaded_ __GHC__ ensures safe FFI calls run in separate OS thread. 
- [__forkOn__](http://hackage.haskell.org/package/base-4.7.0.0/docs/Control-Concurrent.html#v:forkOn) lets you run on a specific CPU to run on - overriding the scheduler. 

#Asynchronus Exceptions 

~~~{.haskell} 
modifyMVar :: MVar a -> (a -> IO (a,b)) -> IO b
modifyMVar m action = do
  v0 <- takeMVar m -- -------------- oops, race condition
  (v, r) <- action v0 `onException` putMVar m v0
  putMVar m v
  return r
~~~

- `modifyMVar` is a utility for updating a  value 
- equivalent of "x++ " in C 
- Notes about its implementation 
	- take old value 
	- exec action , gets new value plus type 
	- if exception, shove old value back into the MVar
	- otherwise put new value 
- problem: really bad place to get `throwTo` or `killThread` call is right after takeMvar and before the `onException`
- Now you've received an Exception but left mvar empty
- _timeout_ function from few slides ago also suffers from the same problem  

#Masking Exceptions
- A solution to the above problem is to use [__mask__](http://hackage.haskell.org/package/base-4.7.0.0/docs/Control-Exception.html#v:mask) 

~~~
mask :: ((forall a. IO a -> IO a) -> IO b) -> IO b 
~~~ 
- ignore the `forall` for now. 
- `mask` allows us to ignore _exceptions_ for a while 
- Exceptions are __automatically unmasked__ if the __thread sleeps__ ( Eg: when blocked on a takeMVar)
- Fixed code for `modifyMVar`

~~~{.haskell} 
modifyMVar :: MVar a -> (a -> IO (a,b)) -> IO b
modifyMVar m action = mask $ \unmask -> do
  v0 <- takeMVar m -- automatically unmasked while waiting
  (v, r) <- unmask (action v0) `onException` putMVar m v0
  putMVar m v
  return r
~~~

- `unmasked` function passed into _mask_ allows exceptions to be unmasked again for an action. After the statement is executed we are back to _masked_ exceptions

#### Masking Exceptions Continued 

- Correct way to wrap actions from a bounded OS thread in an unbounded Thread

~~~{.haskell} 
wrap :: IO a -> IO a          -- Fixed version of wrap
wrap action = do
  mv <- newEmptyMVar
  mask $ \unmask -> do
      tid <- forkIO $ (unmask action >>= putMVar mv) `catch`
                      \e@(SomeException _) -> putMVar mv (throw e)
      let loop = takeMVar mv `catch` \e@(SomeException _) ->
                 throwTo tid e >> loop
      loop
~~~

- `forkIO` preserves the __current masked state.__ 
- Therefore we can use _unmask_ in a child thread ( as above)
- new child thread starts of in the masked state
- unmask exceptions during the  execution of an action
- if parent gets the exception - pass it to the child and 
		if we get it back then its fine, repeat until takeMvar succeeds

#The Bracket Function
- `mask` is tricky but [__bracket__](http://hackage.haskell.org/package/base-4.7.0.0/docs/Control-Exception.html#v:bracket) makes it easy to use. 
~~~{.haskell} 
bracket :: IO a -> (a -> IO b) -> (a -> IO c) -> IO c  
~~~  
- `bracket` takes as its arguments 
	+ An initial function
	+ A cleanup function
	+ The main action 
	+ Result is return of the main action

Following is an Example: 

~~~{.haskell} 
bracket (openFile "/etc/mtab" ReadMode) -- first
        hClose                          -- last
        (\h -> hGetContents h >>= doit) -- main
~~~
- No matter what happens ( synchornus or asynchronus exceptions ) we are `guaranteed` to __close the file handle__.
- Source code for bracket (from hoogle)
~~~{.haskell} 
 bracket
         :: IO a         -- ^ computation to run first (\"acquire resource\")
         -> (a -> IO b)  -- ^ computation to run last (\"release resource\")
         -> (a -> IO c)  -- ^ computation to run in-between
         -> IO c         -- returns the value from the in-between computation
 bracket before after thing =
   mask $ \restore -> do
     a <- before
     r <- restore (thing a) `onException` after a
     _ <- after a
     return r

~~~ 

#MVar Applications
####Mutex with MVar 

- Use a `full MVar` to signify that a thread has a lock 
- `Empty` to signify unlocked
	+ Why? 
		* If we switched the order around(i:e `empty = locked` and `full = unlocked`) then any thread can unlock and not just the one that acquired the lock. 

Code for Mutex using __MVars__
~~~{.haskell} 
type Mutex = MVar ThreadId

mutex_create :: IO Mutex
mutex_create = newEmptyMVar

mutex_lock, mutex_unlock :: Mutex -> IO ()

mutex_lock mv = myThreadId >>= putMVar mv

mutex_unlock mv = do mytid <- myThreadId
                     lockTid <- tryTakeMVar mv
                     unless (lockTid == Just mytid) $
                         error "mutex_unlock"
~~~


- Putting __MVars inside MVars__ is a very powerful idea. 
	+ Can be used to implement a [__Condition Variable__](http://www.scs.stanford.edu/14sp-cs240h/slides/concurrency-slides.html#(26))

####Channels using MVar 
- Nesting MVar's can also be used to implement Channels 
	+ Implemented as `two MVars` one for __read__ and the other for __writing__ to end of *Stream* 
	+ one mVar points to head of the list for readers 
	+ one points for the end of the list for writers 
	+ Essentially a linked list of MVars 
	+ Helpful [__Figure__](http://www.scs.stanford.edu/14sp-cs240h/slides/concurrency-slides.html#(27)) for visualization 

~~~{.haskell} 
data Item a = Item a (Stream a)
type Stream a = MVar (Item a)
data Chan a = Chan (MVar (Stream a)) (MVar (Stream a))

newChan :: IO (Chan a)
newChan = do
  empty <- newEmptyMVar -- empty has type (MVar a) which satisfies (MVar (Item a))
  liftM2 Chan (newMVar empty) (newMVar empty)
-- liftM2 is like liftM for functions of two arguments:
-- liftM2 f m1 m2 = do x1 <- m1; x2 <- m2; return (f x1 x2)

writeChan :: Chan a -> a -> IO ()
writeChan (Chan _ w) a = do
  empty <- newEmptyMVar
  modifyMVar_ w $ \oldEmpty -> do
    putMVar oldEmpty (Item a empty)
    return empty

readChan :: Chan a -> IO a
readChan (Chan r _) =
    modifyMVar r $ \full -> do
      (Item a newFull) <- takeMVar full
      return (newFull, a)
~~~

- The above is a simplified implementation 
- [__Control.Concurrent.Chan__](http://hackage.haskell.org/package/base-4.7.0.0/docs/Control-Concurrent-Chan.html) provides *unbounded* Channels

#Networking 
~~~{.haskell} 
connectTo :: HostName -> PortID -> IO Handle
listenOn :: PortID -> IO Socket
accept :: Socket -> (Handle, HostName, PortNumber)
sClose :: Socket -> IO ()
hClose :: Handle -> IO ()
~~~

Support for high-level stream (`TCP` and `Unix-domain`)socket support in [__Network__](http://hackage.haskell.org/package/network-2.5.0.0/docs/Network.html)

#### Example 

__Build a program that plays two users against one another and exits after one game__

~~~{.haskell} 
play :: MVar Move -> MVar Move
     -> (Handle, HostName, PortNumber) -> IO ()
play myMoveMVar opponentMoveMVar (h, host, port) = do
  putStrLn $ "Connection from host " ++ host ++ " port " ++ show port
  myMove <- getMove h
  putMVar myMoveMVar myMove
  opponentMove <- takeMVar opponentMoveMVar
  let o = outcome myMove opponentMove
  hPutStrLn h $ "You " ++ show o

netrock :: PortID -> IO ()
netrock listenPort =
  bracket (listenOn listenPort) sClose $ \s -> do
    mv1 <- newEmptyMVar
    mv2 <- newEmptyMVar
    let cleanup mv (h,_,_) = do
          tryPutMVar mv (error "something blew up")
          hClose h
    wait <- newEmptyMVar
    forkIO $ bracket (accept s) (cleanup mv1) (play mv1 mv2)
      `finally` putMVar wait ()
    bracket (accept s) (cleanup mv2) (play mv2 mv1)
    takeMVar wait
~~~

+ Store the moves for both the players in MVars 
+ Spin of an unbounded thread to handle the incoming connection for `player 1`
+ We use the `wait` __MVar__ to synchronise between the parent thread and the child thread. 
	- If the parent thread executes first then the program blocks on `takeMVar wait` until the child thread finishes execution and runs `putMVar wait ()` 
+ `play` is called once from each thread. If `player1` calls `play` then it __fills__ its MVar and __empties__ the other players ( `player 2` in this case) . 
	+ The use of blocking `takeMVar` helps in synchronising between the two players. 

#### Low Level BSD Socket Support in [__Network.Socket__](http://hackage.haskell.org/package/network-2.5.0.0/docs/Network-Socket.html)

~~~{.haskell} 
socket :: Family -> SocketType -> ProtocolNumber -> IO Socket
connect :: Socket -> SockAddr -> IO ()
bindSocket :: Socket -> SockAddr -> IO ()
listen :: Socket -> Int -> IO ()
accept :: Socket -> IO (Socket, SockAddr)
getAddrInfo :: Maybe AddrInfo
            -> Maybe HostName -> Maybe ServiceName
            -> IO [AddrInfo]
~~~

+ [__netcat Example__](http://www.scs.stanford.edu/14sp-cs240h/slides/concurrency-slides.html#(34))










